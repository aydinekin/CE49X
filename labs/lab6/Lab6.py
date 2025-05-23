import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re

from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB


def main():
    # Load the dataset (replace with pd.read_excel if using the *.xlsx* file)
    with open("construction_documents.json", "r") as file:
        data = json.load(file)

    # Check for missing values
    missing_values = check_for_missing_values(data)
    if any(missing_values.values()):
        print("Missing values found in the dataset.")
        data = handle_missing_values_with_most_frequent(data)
        # data = handle_missing_values_with_constant(data)
    else:
        print("No missing values found in the dataset.")

    # Apply preprocessing to the 'content' field of each document
    for document in data:
        document["content"] = preprocess_construction_text(document["content"])

    # Extract relevant metadata
    metadata = [
        {
            "project_phase": doc["project_phase"],
            "author_role": doc["author_role"],
        }
        for doc in data
    ]
    document_type = [doc["document_type"] for doc in data]
    content = [doc["content"] for doc in data]

    # One‑Hot Encoding using DictVectorizer
    vectorizer = DictVectorizer()  # sparse=True is default and preferred
    encoded_metadata = vectorizer.fit_transform(metadata)

    # --- CLASSIFIER #1: predict *document_type* --------------------------------
    handle_naive_bayes_classification(
        encoded_metadata, content, document_type, "Document Type"
    )
    print("\n //////////////////////////////////////////////////// \n")

    # --- CLASSIFIER #2: predict *project_phase* --------------------------------
    handle_metadata_target_classification(data, target="project_phase")
    print("\n //////////////////////////////////////////////////// \n")

    # Convert to DataFrame for easier handling
    encoded_df = pd.DataFrame(
        encoded_metadata.toarray(), columns=vectorizer.get_feature_names_out()
    )
    encoded_df["document_type"] = document_type
    encoded_df["project_phase"] = [m["project_phase"] for m in metadata]

    # Aggregate data for heat‑map
    distribution = (
        encoded_df.groupby(["project_phase", "document_type"]).size().unstack(fill_value=0)
    )

    # Visualise the distribution heat‑map
    plt.figure(figsize=(12, 6))
    sns.heatmap(distribution, annot=True, fmt="d", cmap="Blues")
    plt.title("Distribution of Document Types Across Project Phases")
    plt.xlabel("Document Type")
    plt.ylabel("Project Phase")
    plt.tight_layout()  #avoids label cut‑off
    plt.show()

    # overall class‑frequency bar chart
    # quick one‑liner requested
    plt.figure(figsize=(8, 4))
    pd.Series(document_type).value_counts().plot.bar()
    plt.title("Overall Document‑Type Frequencies")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()


def handle_naive_bayes_classification(encoded_metadata, contents, targets, title):
    print(f"Training Naive Bayes for: {title}")

    # TF‑IDF vectorisation – we now return both features and the vectoriser
    X_text, tfidf_vectorizer = extract_features_tfidf_vectorizer(contents)

    # Concatenate sparse metadata + sparse text matrix (hstack keeps sparsity)
    from scipy import sparse

    X = sparse.hstack([encoded_metadata, X_text])
    y = targets

    # Train / test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train classifier
    clf = MultinomialNB()
    clf.fit(X_train, y_train)

    # Report accuracy
    score = clf.score(X_test, y_test)
    print(f"Accuracy: {score:.3f}")

    # Full evaluation & plots
    evaluate_model(
        clf,
        X_test,
        y_test,
        clf.classes_,
        tfidf_vectorizer=tfidf_vectorizer,
    )
    print("Evaluation is done.\n")


# -----------------------------------------------------------------------------
#  Two quick wrappers for *project_phase*
# -----------------------------------------------------------------------------

def handle_metadata_target_classification(data, target):
    # Build metadata minus the column we want to predict
    other_meta_keys = {
        "project_phase",
        "author_role",
        "document_type",
    } - {target}

    metadata = [
        {k: doc[k] for k in other_meta_keys} for doc in data
    ]
    y = [doc[target] for doc in data]
    content = [doc["content"] for doc in data]

    vectorizer = DictVectorizer()
    encoded_metadata = vectorizer.fit_transform(metadata)

    handle_naive_bayes_classification(
        encoded_metadata, content, y, title=target.replace("_", " ").title()
    )

def evaluate_model(clf, X_test, y_test, target_names, tfidf_vectorizer=None):

    y_pred = clf.predict(X_test)
    labels = list(target_names)
    print("Classification report:")
    print(classification_report(y_test, y_pred, labels=labels, target_names=labels, zero_division=0))

    cm = confusion_matrix(y_test, y_pred, labels=labels)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.show()

    print("Most frequent confusions (off‑diagonal counts > 0):")
    for i, row in enumerate(cm):
        for j, val in enumerate(row):
            if i != j and val > 0:
                print(f"  {labels[i]} ↔ {labels[j]} : {val}x")

    # Top‑10 TF‑IDF terms per class (auto‑detect metadata offset)
    if tfidf_vectorizer is not None:
        feature_names = np.array(tfidf_vectorizer.get_feature_names_out())
        # metadata columns = total features − tfidf features
        meta_offset = clf.feature_log_prob_.shape[1] - len(feature_names)
        print("Top‑10 discriminative TF‑IDF terms per class:")
        for i, class_label in enumerate(clf.classes_):
            log_probs_text = clf.feature_log_prob_[i][meta_offset:]
            top10 = np.argsort(log_probs_text)[-10:][::-1]
            print(f"  {class_label}: {', '.join(feature_names[top10])}")


def extract_features_tfidf_vectorizer(contents):
    vec = TfidfVectorizer()
    return vec.fit_transform(contents), vec


def extract_features_count_vectorizer(contents):
    vec = CountVectorizer()
    return vec.fit_transform(contents), vec



def preprocess_construction_text(text):
    # 1. Abbreviation expansion
    abbreviations = {
        "MEP": "Mechanical Electrical Plumbing",
        "PPE": "Personal Protective Equipment",
        "RFI": "Request for Information",
        "CO": "Change Order",
        "SI": "Safety Incident",
        "QI": "Quality Inspection",
    }
    for abbr, full_form in abbreviations.items():
        text = re.sub(rf"\b{abbr}\b", full_form, text)

    # 2. Unit normalisation
    text = re.sub(r"(\d+)(m|cm|mm|kg|tons?)\b", r"\1 \2", text)
    text = re.sub(r"\b(\d+)\s?(m)\b", r"\1 meters", text)
    text = re.sub(r"\b(\d+)\s?(kg)\b", r"\1 kilograms", text)

    # 3. Terminology normalisation
    terminology = {
        "confined space": "restricted area",
        "structural connection": "structural joint",
        "finishing work": "final construction tasks",
    }
    for term, standard_term in terminology.items():
        text = re.sub(rf"\b{term}\b", standard_term, text, flags=re.IGNORECASE)

    return text.lower()  # lower‑case for vectoriser consistency


# TF‑IDF matrix and the fitted vectoriser

def extract_features_tfidf_vectorizer(contents):
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_features = tfidf_vectorizer.fit_transform(contents)
    return tfidf_features, tfidf_vectorizer


#tfidf vectorizer used but count vectorizer is also demonstrated below

def extract_features_count_vectorizer(contents):
    count_vectorizer = CountVectorizer()
    count_features = count_vectorizer.fit_transform(contents)
    return count_features, count_vectorizer

def check_for_missing_values(data):
    missing_values = {key: sum(1 for doc in data if doc[key] is None) for key in data[0].keys()}
    print("\nMissing Values in Dataset:")
    for key, count in missing_values.items():
        print(f"  {key}: {count} missing values")
    return missing_values

def handle_missing_values_with_most_frequent(data):
    df = pd.DataFrame(data)
    imputer = SimpleImputer(strategy="most_frequent")
    df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
    return df_imputed.to_dict(orient="records")


def handle_missing_values_with_constant(data):
    df = pd.DataFrame(data)
    imputer = SimpleImputer(strategy="constant", fill_value="Unknown")
    df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
    return df_imputed.to_dict(orient="records")


# -----------------------------------------------------------------------------
if __name__ == "__main__":
    main()
