import os
import pandas as pd
from matplotlib import pyplot as plt
from feature_extraction import generate_feature_vectors_from_samples

feature_df = pd.DataFrame()

for root, dirs, files in os.walk('/output_directory/good_data/'):
    for csv_file_path in files:
        print(
            csv_file_path
        )
        df = pd.read_csv(os.path.join(root, csv_file_path))
        df = df.drop('Timestamp', axis=1)
        valence_value = df['Valence'][0]
        arousal_value = df['Arousal'][0]
        dominance_value = df['Dominace'][0]
        df = df.drop('Valence', axis=1)
        df = df.drop('Arousal', axis=1)
        df = df.drop('Dominace', axis=1)
        df = df.drop('Viewed', axis=1)
        df = df.drop('Video', axis=1)
        df = df.drop('Participant', axis=1)

        csv_data = df.to_numpy()

        matrix = csv_data
        vectors, header = generate_feature_vectors_from_samples(matrix, nsamples=150, period=3., state=None)

        features_reshaped = vectors.reshape(-1, vectors.shape[-1])

        # Create DataFrame for features
        features_df = pd.DataFrame(features_reshaped,
                                   columns=[f"Feature_{i}" for i in range(features_reshaped.shape[1])])

        # Add valence, arousal, and dominance values as separate columns
        features_df['Valence'] = valence_value
        features_df['Arousal'] = arousal_value
        features_df['Dominance'] = dominance_value

        # Append features_df to feature_df
        feature_df = pd.concat([feature_df, features_df], ignore_index=True)

    # Display the resulting DataFrame
    print(feature_df)

feature_df.to_csv('features.csv', index=False)

# Checking for imbalanced classes
valence_counts = feature_df['Valence'].value_counts()
arousal_counts = feature_df['Arousal'].value_counts()
dominance_counts = feature_df['Dominance'].value_counts()

# Plotting Valence Frequency
plt.bar(valence_counts.index, valence_counts.values)
plt.xlabel('Valence')
plt.ylabel('Frequency')
plt.title('Frequency of Valence Values')
plt.xticks(valence_counts.index)
plt.show()

# Plotting Arousal Frequency
plt.bar(arousal_counts.index, arousal_counts.values)
plt.xlabel('Arousal')
plt.ylabel('Frequency')
plt.title('Frequency of Arousal Values')
plt.xticks(arousal_counts.index)
plt.show()

# Plotting Dominance Frequency
plt.bar(dominance_counts.index, dominance_counts.values)
plt.xlabel('Dominance')
plt.ylabel('Frequency')
plt.title('Frequency of Dominance Values')
plt.xticks(dominance_counts.index)
plt.show()