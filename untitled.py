import evaluate
rouge_metric = evaluate.load("rouge")
print(rouge_metric)