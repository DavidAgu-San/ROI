import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

# Generate sample customer data (replace this with your actual data loading)
def generate_sample_data(n_samples=100):
    """Generate sample customer data for demonstration"""
    data = {
        'customer_id': range(1, n_samples + 1),
        'total_spend': np.random.uniform(10, 5000, n_samples),
        'avg_order_value': np.random.uniform(20, 500, n_samples),
        'order_frequency': np.random.uniform(1, 30, n_samples),
        'days_since_last_order': np.random.uniform(0, 180, n_samples),
        'payment_reliability': np.random.choice(['Reliable', 'Unreliable'], n_samples, p=[0.6, 0.4])
    }
    return pd.DataFrame(data)

# Load or generate data
print("Loading customer data...")
# Replace this with: df = pd.read_csv('your_data.csv')
df = generate_sample_data(100)

# Prepare features for clustering
feature_columns = ['total_spend', 'avg_order_value', 'order_frequency', 'days_since_last_order']
X_features = df[feature_columns].copy()

# Standardize features for K-means
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_features)

# Apply K-means clustering for customer segmentation
print("\nApplying K-means clustering...")
n_clusters = 3  # Adjust based on your business needs
kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
df['cluster'] = kmeans.fit_predict(X_scaled)

# Add cluster as a feature for the classification model
X_features['cluster'] = df['cluster']

# Prepare target variable
le = LabelEncoder()
y = le.fit_transform(df['payment_reliability'])

# Prepare features for Random Forest
X = X_features.copy()

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Train Random Forest Classifier
print("\nTraining Random Forest Classifier...")
rf_classifier = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    random_state=42,
    class_weight='balanced'
)
rf_classifier.fit(X_train, y_train)

# Make predictions
y_pred = rf_classifier.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy: {accuracy:.2f}")

# Generate predictions for all customers
df['payment_reliability_pred'] = le.inverse_transform(rf_classifier.predict(X))

# Create standpoint classification based on total_spend and payment_reliability
def assign_standpoint(row):
    spend = row['total_spend']
    reliability = row['payment_reliability_pred']
    
    if reliability == 'Unreliable':
        if spend < 500:
            return 'Risky Low/Medium Spender'
        else:
            return 'Risky High Spender'
    else:
        if spend < 500:
            return 'Safe Low/Medium Spender'
        else:
            return 'Safe High Spender'

df['standpoint'] = df.apply(assign_standpoint, axis=1)

# Create customer profiles DataFrame
customer_profiles = df[['customer_id', 'total_spend', 'cluster', 
                        'payment_reliability_pred', 'standpoint']].copy()
customer_profiles.columns = ['customer_id', 'total_spend', 'cluster', 
                              'payment_reliability', 'standpoint']

# Display customer profiles
print("\n" + "="*80)
print("Customer Profiles:")
print("="*80)
print(customer_profiles.head(10).to_string(index=False))
print(f"\n[{len(customer_profiles)} rows x {len(customer_profiles.columns)} columns]")

# Generate classification report
print("\n" + "="*80)
print("Payment Reliability Classification Report:")
print("="*80)
target_names = le.classes_
print(classification_report(y_test, y_pred, target_names=target_names))

# Feature importance
print("\n" + "="*80)
print("Feature Importance:")
print("="*80)
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': rf_classifier.feature_importances_
}).sort_values('importance', ascending=False)
print(feature_importance.to_string(index=False))

# Save results
customer_profiles.to_csv('customer_profiles.csv', index=False)
print("\nCustomer profiles saved to 'customer_profiles.csv'")

