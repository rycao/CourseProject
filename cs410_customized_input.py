from nltk.stem import PorterStemmer
import pandas as pd
import pickle
import joblib
from nltk.corpus import stopwords
import warnings
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)
en_stop = set(stopwords.words('english'))

loaded_model_info = joblib.load('cs410_final_mmodel_info.joblib')

while True:
    pos_review = input("Positive Review: ")
    neg_review = input("Negative Review: ")
    if pos_review == "" and neg_review == "":
        print("bye")
        break
    in_str = [pos_review, neg_review]
    single_X = pd.DataFrame([in_str], columns=['Negative_Review', 'Positive_Review'])

    # Remove stopwords
    single_X['Positive_Review'] = single_X['Positive_Review'].apply(
        lambda x: ' '.join([w for w in x.strip().lower().split() if w not in en_stop]))
    single_X['Negative_Review'] = single_X['Negative_Review'].apply(
        lambda x: ' '.join([w for w in x.strip().lower().split() if w not in en_stop]))

    # Stemming
    ps = PorterStemmer()
    single_X['Positive_Review'] = single_X['Positive_Review'].apply(
        lambda x: ' '.join([ps.stem(y) for y in x.split()]))
    single_X['Negative_Review'] = single_X['Negative_Review'].apply(
        lambda x: ' '.join([ps.stem(y) for y in x.split()]))

    for ft in loaded_model_info["features"]:
        is_p = ft[0][8] == "p"
        f = ' '.join(ft[0][10:].split("__"))
        if is_p:
            single_X["feature_p_" + "__".join(f.split())] = single_X['Positive_Review'].apply(lambda x: f in x)
        else:
            single_X["feature_n_" + "__".join(f.split())] = single_X['Negative_Review'].apply(lambda x: f in x)

    single_X = single_X.drop(["Positive_Review", "Negative_Review"], 1)

    print("Our best estimate of your rating is:", loaded_model_info["model"].predict(single_X)[0])