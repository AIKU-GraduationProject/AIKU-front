import logging
from datasets import load_metric
from datasets import Dataset

from transformers import AutoConfig, AutoModelForQuestionAnswering, AutoTokenizer

from transformers import (
    DataCollatorWithPadding,
    EvalPrediction,
    TrainingArguments,
)

from models.mrc.utils_qa import postprocess_qa_predictions
from models.mrc.trainer_qa import QuestionAnsweringTrainer

from models.mrc.arguments import (
    ModelArguments,
    DataTrainingArguments,
)

logger = logging.getLogger(__name__)

import pandas as pd


model_args = ModelArguments(
    config_name=None,
    model_name_or_path="/home/konkuk/Desktop/koo/MRC/models",
    tokenizer_name=None,
)

data_args = DataTrainingArguments(
    #dataset_name='./data/train_dataset',
    overwrite_cache=False,
    preprocessing_num_workers=None,
    max_seq_length=384,
    pad_to_max_length=False,
    doc_stride=128,
    max_answer_length=30,
    train_retrieval=False,
    eval_retrieval=False,
    dataset_config_name=None,
    train_file=None,
    validation_file=None
)

training_args = TrainingArguments(
    output_dir="./outputs/klue/roberta-large",
    overwrite_output_dir=True,
    do_eval=True,
    do_predict=True,
    per_device_train_batch_size=3,
)

# Load pretrained model and tokenizer
print("**************************************** MRC Setting ****************************************")
config = AutoConfig.from_pretrained(
    model_args.config_name
    if model_args.config_name
    else model_args.model_name_or_path,
)
tokenizer = AutoTokenizer.from_pretrained(
    model_args.tokenizer_name
    if model_args.tokenizer_name
    else model_args.model_name_or_path,
    #use_fast=True,
)
model = AutoModelForQuestionAnswering.from_pretrained(
    model_args.model_name_or_path,
    from_tf=bool(".ckpt" in model_args.model_name_or_path),
    config=config,
)

def main(query, passge):
    # TODO: input_context 변수 만들어서 리트리벌 결과 선택된 문서 넣기-근영님
    input_context = passge
    input_question = query
    #input_question = input('질문 입력:') # TODO: input 부분


    test = pd.DataFrame({'guid':[20190103],
                    'context':[input_context],
                    'question':
                             [input_question],
                         })

    datasets = Dataset.from_pandas(test)

    #print(datasets)



    # train or eval mrc model
    # TODO: Retrieval 리턴 값 not None이면 실행되도록
    mrc_answer = run_mrc(data_args, training_args, model_args, datasets, tokenizer, model)
    print(f"답은: {mrc_answer}") #test용
    return mrc_answer # TODO: ouput 부분입니담(답 못하면 "no_answer" return 가능하면 답변

def run_mrc(data_args, training_args, model_args, datasets, tokenizer, model):
    column_names = datasets.column_names

    question_column_name = "question"
    context_column_name = "context"
    answer_column_name = "answers"

    # Padding side determines if we do (question|context) or (context|question).
    pad_on_right = tokenizer.padding_side == "right"

    # Validation preprocessing
    def prepare_validation_features(examples):
        # Tokenize our examples with truncation and maybe padding, but keep the overflows using a stride. This results
        # in one example possible giving several features when a context is long, each of those features having a
        # context that overlaps a bit the context of the previous feature.
        tokenized_examples = tokenizer(
            examples[question_column_name if pad_on_right else context_column_name],
            examples[context_column_name if pad_on_right else question_column_name],
            truncation="only_second" if pad_on_right else "only_first",
            max_length=384,
            stride=128,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length" if data_args.pad_to_max_length else False,
        )

        # Since one example might give us several features if it has a long context, we need a map from a feature to
        # its corresponding example. This key gives us just that.
        sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")

        # For evaluation, we will need to convert our predictions to substrings of the context, so we keep the
        # corresponding example_id and we will store the offset mappings.
        tokenized_examples["example_id"] = []

        for i in range(len(tokenized_examples["input_ids"])):
            # Grab the sequence corresponding to that example (to know what is the context and what is the question).
            sequence_ids = tokenized_examples.sequence_ids(i)
            context_index = 1 if pad_on_right else 0

            # One example can give several spans, this is the index of the example containing this span of text.
            sample_index = sample_mapping[i]
            tokenized_examples["example_id"].append(examples["guid"][sample_index])

            # Set to None the offset_mapping that are not part of the context so it's easy to determine if a token
            # position is part of the context or not.
            tokenized_examples["offset_mapping"][i] = [
                (o if sequence_ids[k] == context_index else None)
                for k, o in enumerate(tokenized_examples["offset_mapping"][i])
            ]
        tokenized_examples.pop('token_type_ids')
        return tokenized_examples

    if training_args.do_eval:
        eval_dataset = datasets

        eval_dataset = eval_dataset.map(
            prepare_validation_features,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not data_args.overwrite_cache,
        )

    # Data collator
    # We have already padded to max length if the corresponding flag is True, otherwise we need to pad in the data collator.
    data_collator = (
        DataCollatorWithPadding(
            tokenizer, pad_to_multiple_of=8 if training_args.fp16 else None
        )
    )

    # Post-processing:
    def post_processing_function(examples, features, predictions, training_args):
        # Post-processing: we match the start logits and end logits to answers in the original context.
        predictions = postprocess_qa_predictions(
            examples=examples,
            features=features,
            predictions=predictions,
            max_answer_length=data_args.max_answer_length,
            output_dir=training_args.output_dir,
        )
        # Format the result to the format the metric expects.
        formatted_predictions = [
            {"id": k, "prediction_text": v} for k, v in predictions.items()
        ]
        if training_args.do_predict:
            return formatted_predictions


    # metric = load_metric("squad")

    # def compute_metrics(p: EvalPrediction):
    #     return metric.compute(predictions=p.prediction_text, references=p.label_ids)

    # Initialize our Trainer
    trainer = QuestionAnsweringTrainer(
        model=model,
        args=training_args,
        train_dataset=None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        eval_examples=datasets if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        post_process_function=post_processing_function,
        # compute_metrics=compute_metrics,
    )

    ans = trainer.qa()
    return ans[0]['prediction_text']

if __name__ == "__main__":
    # # Load pretrained model and tokenizer
    # config = AutoConfig.from_pretrained(
    #     model_args.config_name
    #     if model_args.config_name
    #     else model_args.model_name_or_path,
    # )
    # tokenizer = AutoTokenizer.from_pretrained(
    #     model_args.tokenizer_name
    #     if model_args.tokenizer_name
    #     else model_args.model_name_or_path,
    #     # use_fast=True,
    # )
    # model = AutoModelForQuestionAnswering.from_pretrained(
    #     model_args.model_name_or_path,
    #     from_tf=bool(".ckpt" in model_args.model_name_or_path),
    #     config=config,
    # )

    main("", "")
