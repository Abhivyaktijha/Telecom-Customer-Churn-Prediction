import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import classification_report
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from scipy.stats import randint, uniform
import joblib
import os


# ─────────────────────────────────────────────
#  1. LOAD & CLEAN DATA
# ─────────────────────────────────────────────

@st.cache_data
def load_and_clean_data(path=r'D:\project 2026\telecom Customer churn\Customer-Churn (1).csv'):
    churn = pd.read_csv(path)
    new_churn = churn.copy()
    new_churn['TotalCharges'] = pd.to_numeric(new_churn['TotalCharges'], errors='coerce')
    new_churn.dropna(inplace=True)

    bins   = list(range(1, 80, 12))
    labels = ["{0}-{1}".format(i, i + 11) for i in range(1, 72, 12)]
    new_churn['tenure_group'] = pd.cut(
        new_churn['tenure'], bins=bins, right=False, labels=labels
    )
    new_churn.drop(columns=['customerID', 'tenure'], inplace=True)

    return new_churn, bins, labels


# ─────────────────────────────────────────────
#  2. TRAIN OR LOAD MODEL
# ─────────────────────────────────────────────

@st.cache_resource
def get_model(_new_churn):
    model_path = 'best_ada_model.pkl'

    # If model already saved, just load it
    if os.path.exists(model_path):
        model = joblib.load(model_path)
        return model

    # Otherwise train from scratch
    X = _new_churn.drop(columns='Churn')
    y = _new_churn['Churn']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    num_f = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    cat_f = X.select_dtypes(include=['object', 'category']).columns.tolist()

    preprocessor = ColumnTransformer(transformers=[
        ('num', StandardScaler(), num_f),
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_f)
    ])

    full_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('Classifier', AdaBoostClassifier())
    ])

    param_distributions = {
        'Classifier__n_estimators': randint(50, 200),
        'Classifier__learning_rate': uniform(0.01, 2.0),
    }

    random_search = RandomizedSearchCV(
        estimator=full_pipeline,
        param_distributions=param_distributions,
        n_iter=50,
        cv=5,
        scoring='f1',
        verbose=0,
        random_state=42,
        n_jobs=-1
    )
    random_search.fit(X_train, y_train)

    best_model = random_search.best_estimator_
    joblib.dump(best_model, model_path)

    return best_model


# ─────────────────────────────────────────────
#  3. PREPROCESS USER INPUT
# ─────────────────────────────────────────────

def preprocess_input(user_input, bins, labels):
    df_in = pd.DataFrame([user_input])
    df_in['tenure_group'] = pd.cut(
        df_in['tenure'], bins=bins, labels=labels, include_lowest=True
    )
    df_in = df_in.drop(columns=['tenure'])
    return df_in


# ─────────────────────────────────────────────
#  4. MAIN APP
# ─────────────────────────────────────────────

def main():
    st.set_page_config(page_title='Telecom Churn Prediction', layout='centered')
    st.title('📡 Telecom Customer Churn Prediction')
    st.markdown('---')

    new_churn, bins, labels = load_and_clean_data()

    with st.spinner('Loading model...'):
        model = get_model(new_churn)

    sample = new_churn
    st.subheader('Enter Customer Details')

    with st.form('input_form'):
        tenure = st.slider('Tenure (months)', min_value=1, max_value=72, value=12)
        user_input = {'tenure': tenure}

        cols_to_ask = [
            c for c in sample.columns
            if c not in ['Churn', 'tenure', 'tenure_group']
        ]

        col1, col2 = st.columns(2)
        for idx, col in enumerate(cols_to_ask):
            target = col1 if idx % 2 == 0 else col2
            with target:
                col_dtype = str(sample[col].dtype)
                if col_dtype in ('object', 'category'):
                    opts = sorted(sample[col].dropna().unique().tolist())
                    user_input[col] = st.selectbox(col, opts)
                else:
                    minv = float(sample[col].min())
                    maxv = float(sample[col].max())
                    defv = float(sample[col].median())
                    if pd.api.types.is_integer_dtype(sample[col]):
                        user_input[col] = st.number_input(
                            col, min_value=int(minv), max_value=int(maxv),
                            value=int(defv), step=1
                        )
                    else:
                        user_input[col] = st.number_input(
                            col, min_value=minv, max_value=maxv,
                            value=defv, step=0.01
                        )

        submitted = st.form_submit_button('🔍 Predict')

    if submitted:
        X_in = preprocess_input(user_input, bins, labels)

        pred_proba = model.predict_proba(X_in)[0][1]
        pred_class = model.predict(X_in)[0]

        st.markdown('---')
        st.subheader('Prediction Result')

        churn_label = 'Yes ⚠️' if pred_class == 1 else 'No ✅'
        m1, m2 = st.columns(2)
        m1.metric('Churn Prediction', churn_label)
        m2.metric('Churn Probability', f'{pred_proba:.1%}')

        if pred_proba >= 0.7:
            st.error('🔴 High risk of churn. Consider retention actions.')
        elif pred_proba >= 0.4:
            st.warning('🟡 Moderate churn risk. Worth monitoring.')
        else:
            st.success('🟢 Low churn risk. Customer appears stable.')

        with st.expander('🛠 Debug info'):
            st.write('**Raw user input:**', user_input)
            st.write('**Preprocessed DataFrame sent to model:**')
            st.dataframe(X_in)

    st.markdown('---')
    st.caption(
        'Model: AdaBoostClassifier tuned with RandomizedSearchCV · '
        'Preprocessing: StandardScaler + OneHotEncoder via ColumnTransformer'
    )


if __name__ == '__main__':
main()