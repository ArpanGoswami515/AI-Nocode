import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, r2_score
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import pickle
from sklearn.impute import SimpleImputer
import copy

# Set Streamlit theme
st.set_page_config(page_title="EDA & Modeling", layout="wide")

def load_and_view_data():
    st.markdown("""
        <style>
            .stApp { background-color: #004953; }
            .stSidebar { background-color: #004040 !important; color: white; }
            .stTitle { color: #007bff; }
        </style>
    """, unsafe_allow_html=True)
    st.title("📂 Load and View Dataset")
    uploaded_file = st.file_uploader("📤 Upload a CSV file", type=["csv"], help="Upload your dataset here.")
    if uploaded_file is not None:
        try:
            st.session_state.df = pd.read_csv(uploaded_file)
            st.success("✅ File uploaded successfully!")
        except Exception as e:
            st.error(f"❌ Error loading file: {e}")
    
    if "df" in st.session_state:
        st.dataframe(st.session_state.df, use_container_width=True)
    else:
        st.warning("⚠️ No dataset loaded.")

def preprocess_data():
    st.markdown("""
        <style>
            .stApp { background-color: #203838; }
            .stSidebar { background-color: #004040 !important; color: white; }
            .stTitle { color: #007bff; }
        </style>
    """, unsafe_allow_html=True)
    st.title("⚙️ Preprocess Dataset")
    if "df" not in st.session_state:
        st.warning("⚠️ Please upload a dataset first.")
        return
    
    columns = st.session_state.df.columns.tolist()
    drop_columns = st.multiselect("🗑️ Select columns to drop", columns)
    label_encode_columns = st.multiselect("🔤 Select categorical columns to encode", st.session_state.df.select_dtypes(include=['object']).columns.difference(drop_columns))
    missing_value_strategy = st.selectbox("🛠️ Handle missing values", ["Mean", "Median", "Mode", "Zero", "Drop"])
    apply_scaling = st.multiselect("📏 Select numerical columns to scale", st.session_state.df.select_dtypes(include=['number']).columns.difference(drop_columns))
    scaling_type = st.selectbox("📊 Scaling type", ["Standard Scaling", "Min-Max Scaling"])
    
    preprcessing_parameters = {
        "drop_columns": drop_columns,
        "label_encode_columns": label_encode_columns,
        "missing_value_strategy": missing_value_strategy,
        "columns_to_apply_scaling": apply_scaling,
        "scaling_type": scaling_type,
        # encoders
        # inputer : for missing value fill (if not drop row)
        # scalars
    }

    if st.button("🚀 Apply Preprocessing"):
        with st.spinner("Processing..."):
            
            encoder = {}
            if drop_columns:
                st.session_state.df.drop(columns=drop_columns, inplace=True)
            
            if label_encode_columns:
                le = LabelEncoder()
                for col in label_encode_columns:
                    st.session_state.df[col] = le.fit_transform(st.session_state.df[col])
                    encoder[col] = copy.deepcopy(le)
                # transformers.append(('label_encoder', le, label_encode_columns))
                preprcessing_parameters["encoders"] = encoder

            transformers = []
            if missing_value_strategy == "Mean":
                imputer = SimpleImputer(strategy='mean')
            elif missing_value_strategy == "Median":
                imputer = SimpleImputer(strategy='median')
            elif missing_value_strategy == "Mode":
                imputer = SimpleImputer(strategy='most_frequent')
            elif missing_value_strategy == "Zero":
                imputer = SimpleImputer(strategy='constant', fill_value=0)
            elif missing_value_strategy == "Drop":
                st.session_state.df.dropna(inplace=True)
            else:
                imputer = SimpleImputer(strategy='mean')  # Default to mean if no strategy is selected
            if missing_value_strategy != "Drop":
                preprcessing_parameters['imputer'] = imputer

            # apply the imputer on data

            if missing_value_strategy != "Drop":
                st.session_state.df = pd.DataFrame(imputer.fit_transform(st.session_state.df), columns=st.session_state.df.columns)
            else:
                st.session_state.df = st.session_state.df.dropna()
            
            st.session_state.df.drop_duplicates(inplace=True)
            scaler={}
            if apply_scaling:
                scalar = StandardScaler() if scaling_type == "Standard Scaling" else MinMaxScaler()
                for i in apply_scaling:
                    st.session_state.df[i] = scalar.fit_transform(st.session_state.df[[i]])
                    scaler[i] = copy.deepcopy(scalar)
                preprcessing_parameters['scalers'] = scaler
                      
            st.success("✅ Preprocessing applied successfully!")
            st.dataframe(st.session_state.df, use_container_width=True)
            
            st.download_button(
                label="📥 Download Preprocessing Pipeline",
                data=pickle.dumps(preprcessing_parameters),
                file_name='preprocessing_pipeline.pkl',
                mime='application/octet-stream'
            )

def visualize_data():
    st.markdown("""
        <style>
            .stApp { background-color:rgb(83, 40, 0); }
            .stSidebar { background-color: #004040 !important; color: white; }
            .stTitle { color: #007bff; }
        </style>
    """, unsafe_allow_html=True)
    st.title("📊 Visualize Dataset")
    if "df" not in st.session_state:
        st.warning("⚠️ Please upload and preprocess the dataset first.")
        return
    
    if st.checkbox("📌 Show Dataset Summary"):
        st.write(st.session_state.df.describe())
        st.download_button(
            label="📥 Download Preprocessed Dataset",
            data=st.session_state.df.to_csv(index=False).encode('utf-8'),
            file_name='preprocessed_dataset.csv',
            mime='text/csv'
        )
    
    if st.checkbox("📊 Show Correlation Heatmap"):
        if st.button("🎨 Generate Heatmap"):
            fig, ax = plt.subplots(figsize=(12, 6))
            sns.heatmap(st.session_state.df.corr(), annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
            st.pyplot(fig)
    
    plot_type = st.selectbox("📈 Select Plot Type", ["Histogram", "Box Plot"])
    selected_column = st.selectbox("📌 Select Column", st.session_state.df.columns)
    
    if st.button("📊 Generate Plot"):
        fig, ax = plt.subplots()
        if plot_type == "Histogram":
            sns.histplot(st.session_state.df[selected_column], kde=True, ax=ax)
        elif plot_type == "Box Plot":
            sns.boxplot(y=st.session_state.df[selected_column], ax=ax)
        elif plot_type == "Scatter Plot":
            scatter_x = st.selectbox("Select X-axis column", st.session_state.df.columns)
            sns.scatterplot(x=st.session_state.df[scatter_x], y=st.session_state.df[selected_column], ax=ax)
        st.pyplot(fig)
    
    plot_type_2d = st.selectbox("📈 Select 2D Plot Type", ["Scatter Plot", "Line Plot"])
    x_column = st.selectbox("📌 Select X-axis Column", st.session_state.df.columns)
    y_column = st.selectbox("📌 Select Y-axis Column", st.session_state.df.columns)
    
    if st.button("📊 Generate 2D Plot"):
        fig, ax = plt.subplots()
        if plot_type_2d == "Scatter Plot":
            sns.scatterplot(x=st.session_state.df[x_column], y=st.session_state.df[y_column], ax=ax)
        elif plot_type_2d == "Line Plot":
            sns.lineplot(x=st.session_state.df[x_column], y=st.session_state.df[y_column], ax=ax)
        st.pyplot(fig)

    if st.checkbox("🔢 Apply PCA"):
        num_components = st.slider("Select Number of Principal Components", min_value=1, max_value=min(len(st.session_state.df.columns), 10), value=2)
        pca = PCA(n_components=num_components)
        pca_data = pca.fit_transform(st.session_state.df.select_dtypes(include=['number']))
        explained_variance = pca.explained_variance_ratio_
        
        st.write("🔍 Explained Variance Ratio:", explained_variance)
        fig, ax = plt.subplots()
        ax.bar(range(1, num_components + 1), explained_variance)
        ax.set_xlabel("Principal Component")
        ax.set_ylabel("Variance Explained")
        ax.set_title("PCA Component Importance")
        st.pyplot(fig)
        
        if num_components == 2:
            fig, ax = plt.subplots()
            ax.scatter(pca_data[:, 0], pca_data[:, 1])
            ax.set_xlabel("PC1")
            ax.set_ylabel("PC2")
            ax.set_title("PCA Projection")
            st.pyplot(fig)

def model_fitting():
    st.markdown("""
        <style>
            .stApp { background-color: #0e2432; }
            .stSidebar { background-color: #004040 !important; color: white; }
            .stTitle { color: #007bff; }
        </style>
    """, unsafe_allow_html=True)
    
    st.title("🤖 Fit Model to Dataset")
    
    if "df" not in st.session_state:
        st.warning("⚠️ Please upload and preprocess the dataset first.")
        return
    
    df = st.session_state.df
    
    st.write("### Preview of Preprocessed Data:", df.head())
    
    label_columns = st.multiselect("Select label columns (optional)", df.columns)
    prediction_type = st.radio("Select prediction type:", ["Classification", "Regression", "Clustering"])
    
    model_options = {
        "Classification": ["Logistic Regression", "SVM", "KNN Classifier", "Decision Tree Classifier"],
        "Regression": ["Linear Regression", "KNN Regressor", "Decision Tree Regressor"],
        "Clustering": ["KMeans"]
    }
    model_name = st.selectbox("Choose model", model_options[prediction_type])
    
    params = {}
    if model_name == "Logistic Regression":
        params['max_iter'] = st.number_input("Max Iterations", min_value=100, max_value=10000, value=1000)
    elif model_name == "SVM":
        params['C'] = st.number_input("C (Regularization parameter)", min_value=0.01, max_value=100.0, value=1.0)
        params['kernel'] = st.selectbox("Kernel", ["linear", "poly", "rbf", "sigmoid"])
    elif model_name == "KNN Classifier" or model_name == "KNN Regressor":
        params['n_neighbors'] = st.number_input("Number of Neighbors", min_value=1, max_value=50, value=5)
    elif model_name == "Decision Tree Classifier" or model_name == "Decision Tree Regressor":
        params['max_depth'] = st.number_input("Max Depth", min_value=1, max_value=100, value=10)
    elif model_name == "KMeans":
        params['n_clusters'] = st.number_input("Number of Clusters", min_value=2, max_value=10, value=3)
    
    if st.button("Train Model"):
        X = df.drop(columns=label_columns) if label_columns else df.iloc[:, :-1]
        y = df[label_columns] if label_columns else df.iloc[:, -1]
        
        if prediction_type in ["Classification", "Regression"]:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y if prediction_type == "Classification" else None
            )
            
            models = {
                "Logistic Regression": LogisticRegression(max_iter=params.get('max_iter', 1000)),
                "SVM": SVC(C=params.get('C', 1.0), kernel=params.get('kernel', 'rbf')),
                "KNN Classifier": KNeighborsClassifier(n_neighbors=params.get('n_neighbors', 5)),
                "Decision Tree Classifier": DecisionTreeClassifier(max_depth=params.get('max_depth', 10)),
                "Linear Regression": LinearRegression(),
                "KNN Regressor": KNeighborsRegressor(n_neighbors=params.get('n_neighbors', 5)),
                "Decision Tree Regressor": DecisionTreeRegressor(max_depth=params.get('max_depth', 10))
            }
            
            model = models[model_name]
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            st.write(f"### {model_name} Results")
            if prediction_type == "Classification":
                st.write(f"**Accuracy**: {accuracy_score(y_test, y_pred) * 100:.2f}%")
                st.text(classification_report(y_test, y_pred))
            else:
                st.write(f"**MSE**: {mean_squared_error(y_test, y_pred):.2f}")
                st.write(f"**R² Score**: {r2_score(y_test, y_pred):.2f}")
        
        elif prediction_type == "Clustering":
            model = KMeans(n_clusters=params.get('n_clusters', 3), random_state=42)
            y_pred = model.fit_predict(X)
            st.write(f"### {model_name} Results")
            st.write(f"**Silhouette Score**: {silhouette_score(X, y_pred):.2f}")

        st.download_button(
            label="📥 Download Model",
            data=pickle.dumps(model),
            file_name=f"{model_name.replace(' ', '_').lower()}_model.pkl",
            mime='application/octet-stream'
        )
def test_model():
    st.markdown("""
        <style>
            .stApp { background-color: #8b8000; }
            .stSidebar { background-color: #004040 !important; color: white; }
            .stTitle { color: #007bff; }
        </style>
    """, unsafe_allow_html=True)
    
    st.title("🧪 Test Model on New Data")
    
    uploaded_file = st.file_uploader("📤 Upload a CSV file for testing", type=["csv"], help="Upload your test dataset here.")
    uploaded_model = st.file_uploader("📤 Upload the trained model", type=["pkl"], help="Upload your trained model here.")
    uploaded_pipeline = st.file_uploader("📤 Upload the preprocessing pipeline", type=["pkl"], help="Upload your preprocessing pipeline here.")
    
    if uploaded_file and uploaded_model and uploaded_pipeline:
        try:
            test_df = pd.read_csv(uploaded_file)
            model = pickle.load(uploaded_model)
            preprocessing_params = pickle.load(uploaded_pipeline)
            
            st.session_state.test_df = test_df
            st.session_state.model = model
            st.session_state.preprocessing_params = preprocessing_params
            
            st.success("✅ Files uploaded successfully!")
        except Exception as e:
            st.error(f"❌ Error loading files: {e}")
    
    if "test_df" in st.session_state and "model" in st.session_state and "preprocessing_params" in st.session_state:
        st.write("### Preview of Test Data:", st.session_state.test_df.head())
        
        # Apply preprocessing steps
        preprocessed_test_df = st.session_state.test_df.copy()
        
        # Drop columns
        if 'drop_columns' in st.session_state.preprocessing_params:
            preprocessed_test_df.drop(columns=st.session_state.preprocessing_params['drop_columns'], inplace=True)
        
        # Label encode columns
        if 'encoders' in st.session_state.preprocessing_params:
            for col, encoder in st.session_state.preprocessing_params['encoders'].items():
                preprocessed_test_df[col] = encoder.transform(preprocessed_test_df[col])
        
        # Handle missing values
        if 'imputer' in st.session_state.preprocessing_params:
            imputer = st.session_state.preprocessing_params['imputer']
            preprocessed_test_df = pd.DataFrame(imputer.transform(preprocessed_test_df), columns=preprocessed_test_df.columns)
        
        # Apply scaling
        if 'scalers' in st.session_state.preprocessing_params:
            for col, scaler in st.session_state.preprocessing_params['scalers'].items():
                preprocessed_test_df[col] = scaler.transform(preprocessed_test_df[[col]])
        
        st.write("### Preprocessed Test Data:", preprocessed_test_df.head())
        
        st.write("### Preprocessing Steps Applied:")
        st.write(f"Columns Dropped: {st.session_state.preprocessing_params.get('drop_columns', [])}")
        st.write(f"Label Encoded Columns: {list(st.session_state.preprocessing_params.get('encoders', {}).keys())}")
        st.write(f"Scaled Columns: {list(st.session_state.preprocessing_params.get('scalers', {}).keys())}")
        
        label_columns = st.selectbox("Select label column (optional)", preprocessed_test_df.columns)
        
        if st.button("Test Model"):
            X_test = preprocessed_test_df.drop(columns=label_columns) if label_columns else preprocessed_test_df.iloc[:, :-1]
            y_test = preprocessed_test_df[label_columns] if label_columns else preprocessed_test_df.iloc[:, -1]
            
            y_pred = st.session_state.model.predict(X_test)
            
            st.write(f"### Model Performance on Test Data")
            if isinstance(st.session_state.model, (LogisticRegression, SVC, KNeighborsClassifier, DecisionTreeClassifier)):
                st.write(f"**Accuracy**: {accuracy_score(y_test, y_pred) * 100:.2f}%")
                st.text(classification_report(y_test, y_pred))
            elif isinstance(st.session_state.model, (LinearRegression, KNeighborsRegressor, DecisionTreeRegressor)):
                st.write(f"**MSE**: {mean_squared_error(y_test, y_pred):.2f}")
                st.write(f"**R² Score**: {r2_score(y_test, y_pred):.2f}")
            elif isinstance(st.session_state.model, KMeans):
                st.write(f"**Silhouette Score**: {silhouette_score(X_test, y_pred):.2f}")

def main():
    st.sidebar.title("🚀 Navigation")
    page = st.sidebar.radio("Go to", ["Load and View Dataset", "Preprocess Dataset", "Visualize Dataset", "Fit Model", "Test Model"],
                            index=0, format_func=lambda x: f"📌 {x}")
    
    if page == "Load and View Dataset":
        load_and_view_data()
    elif page == "Preprocess Dataset":
        preprocess_data()
    elif page == "Visualize Dataset":
        visualize_data()
    elif page == "Fit Model":
        model_fitting()
    elif page == "Test Model":
        test_model()

if __name__ == "__main__":
    main()
