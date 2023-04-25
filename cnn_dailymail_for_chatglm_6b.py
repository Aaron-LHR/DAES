import datasets
from datasets import load_dataset


def batch_map_all_subject_verb_obj(examples):
    import random
    prompt_sep_token = "Summary:"
    summary_column = "highlights"
    import spacy
    from nltk.tokenize import sent_tokenize
    # Load the parser
    nlp = spacy.load('en_core_web_sm')

    def get_all_subject_verb_object(sentence):
        """
        Extracts subject, verb and object from sentence using spaCy dependency parser.
        """
        # Parse the sentence
        doc = nlp(sentence)

        # Extract subject, verb and object
        subject = []
        verb = []
        obj = []

        for token in doc:
            if 'subj' in token.dep_:
                subject.append(token.text)
            elif 'obj' in token.dep_:
                obj.append(token.text)
            elif 'ROOT' in token.dep_:
                verb.append(token.text)

        return subject, verb, obj

    def get_subject_verb_obj_new_label(sents):
        subjects, verbs, objs = [], [], []
        for sent in sent_tokenize(sents):
            subject, verb, obj = get_all_subject_verb_object(sent)
            subjects.extend(subject)
            verbs.extend(verb)
            objs.extend(obj)
        # subjects = random.sample(subjects, min(len(subjects), 6))
        # verbs = random.sample(verbs, min(len(verbs), 6))
        # objs = random.sample(objs, min(len(objs), 6))
        res = f"Subjects: {', '.join(subjects)}. Predicate: {', '.join(verbs)}. Object: {', '.join(objs)}. {prompt_sep_token} {sents}"
        return res

    summarys = []
    for summary in examples[summary_column]:
        summarys.append(get_subject_verb_obj_new_label(summary))
    return {summary_column: summarys}

if __name__ == '__main__':
    dataset = load_dataset("cnn_dailymail", '3.0.0')
    sub_dataset = dataset.map(batch_map_all_subject_verb_obj, batched=True, num_proc=15)


    import json

    out = open("ChatGLM-6B\ptuning\data\cnn_dailymail_svo_train.json", "w", encoding="utf-8")
    for item in sub_dataset["train"]:
        out.write(json.dumps({"article": item["article"], "highlights": item["highlights"].replace("\n", " ")}) + "\n")
    out.close()

    out = open("ChatGLM-6B\ptuning\data\cnn_dailymail_svo_test.json", "w", encoding="utf-8")
    for item in sub_dataset["test"]:
        out.write(json.dumps({"article": item["article"], "highlights": item["highlights"].replace("\n", " ")}) + "\n")
    out.close()