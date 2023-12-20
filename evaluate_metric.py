from nltk.translate.bleu_score import corpus_bleu, sentence_bleu, SmoothingFunction

from rouge import Rouge
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from bert_score import BERTScorer,score

scorer = BERTScorer(lang='en',rescale_with_baseline=False)
def get_BLEU(reference, hypothesis):
    reference_paragraph = sent_tokenize(reference)  # list
    candidate_sentence = sent_tokenize(hypothesis)  # list
    reference = list()
    candidate = list()
    for i in reference_paragraph:
        reference.append(word_tokenize(i))

    reference_result = list()
    for j in range(len(candidate_sentence)):
        reference_result.append(reference)

    for k in candidate_sentence:
        candidate.append(word_tokenize(k))
    bleu = corpus_bleu(reference_result, candidate, smoothing_function=SmoothingFunction().method7)
    return bleu


def get_ROUGE(reference, hypothesis):
    rouger = Rouge()
    scores = rouger.get_scores(hypothesis, reference)

    return scores[0]['rouge-l']['f']





def get_bert_score(reference, hypothesis):

    P,R,F1= scorer.score(cands=[hypothesis],refs=[reference])

    return F1
