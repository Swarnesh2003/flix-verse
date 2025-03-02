import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import warnings
warnings.filterwarnings('ignore')

# Define feature names
feature_names = ['mfcc_0', 'mfcc_1', 'mfcc_2', 'mfcc_3', 'mfcc_4', 'mfcc_5', 'mfcc_6', 'mfcc_7', 
                 'mfcc_8', 'mfcc_9', 'mfcc_10', 'mfcc_11', 'mfcc_12', 'zcr', 'rmse', 
                 'spectral_centroid', 'spectral_bandwidth', 'spectral_rolloff',
                 'spectral_contrast_0', 'spectral_contrast_1', 'spectral_contrast_2',
                 'spectral_contrast_3', 'spectral_contrast_4', 'spectral_contrast_5',
                 'spectral_contrast_6', 'chroma_0', 'chroma_1', 'chroma_2', 'chroma_3',
                 'chroma_4', 'chroma_5', 'chroma_6', 'chroma_7', 'chroma_8', 'chroma_9',
                 'chroma_10', 'chroma_11', 'mel_0', 'mel_1', 'mel_2', 'mel_3', 'mel_4',
                 'mel_5', 'mel_6', 'mel_7', 'mel_8', 'mel_9', 'mel_10', 'mel_11', 'mel_12',
                 'tonnetz_0', 'tonnetz_1', 'tonnetz_2', 'tonnetz_3', 'tonnetz_4', 'tonnetz_5',
                 'tempo']

def create_sample_data():
    """
    Create a sample dataset if you don't have one.
    In reality, replace this with your actual data.
    """
    # Create sample data
    np.random.seed(42)
    n_samples = 300
    
    # Generate random features
    data = []
    labels = ['songs', 'fights', 'dialogues']
    
    for _ in range(n_samples):
        # Generate random features
        features = np.random.randn(len(feature_names))
        label = np.random.choice(labels)
        data.append(list(features) + [label])
    
    # Create DataFrame
    df = pd.DataFrame(data, columns=feature_names + ['label'])
    return df

def train_model(df):
    """
    Train the Random Forest model on the provided DataFrame
    """
    print("Training model...")
    
    # Separate features and labels
    X = df[feature_names]
    y = df['label']
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Initialize and train the model
    rf_classifier = RandomForestClassifier(n_estimators=100, 
                                         random_state=42,
                                         n_jobs=-1)
    rf_classifier.fit(X_train, y_train)
    
    # Make predictions on test set
    y_pred = rf_classifier.predict(X_test)
    
    # Print model performance
    print("\nModel Performance:")
    print("=================")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    # Feature importance
    feature_imp = pd.DataFrame({'feature': feature_names, 
                              'importance': rf_classifier.feature_importances_})
    feature_imp = feature_imp.sort_values('importance', ascending=False)
    
    print("\nTop 10 Most Important Features:")
    print(feature_imp.head(10))
    
    # Save the model
    joblib.dump(rf_classifier, 'audio_classifier.joblib')
    print("\nModel saved as 'audio_classifier.joblib'")
    
    return rf_classifier

def predict_audio(features, model=None):
    """
    Predict the class for new audio features
    """
    if model is None:
        model = joblib.load('audio_classifier.joblib')
    
    # Convert features to 2D array
    features_array = np.array(features).reshape(1, -1)
    
    # Make prediction
    prediction = model.predict(features_array)
    probabilities = model.predict_proba(features_array)
    
    # Get probability scores
    class_probabilities = dict(zip(model.classes_, probabilities[0]))
    
    return prediction[0], class_probabilities

def main():
    # Your test sample
    #test_sample = [-57.5206, 79.09011, 6.385821, 17.32118, 5.732647, 9.78912, 1.855125, 7.679437, 4.446654, 5.111172, 4.901502, 5.478502, 4.373956, 0.071714, 0.308703, 2288.746, 2611.765, 5220.243, 20.37801, 13.15327, 17.05031, 16.82359, 18.17652, 16.60293, 15.28199, 0.407342, 0.495072, 0.505526, 0.426909, 0.473899, 0.455802, 0.514613, 0.47325, 0.467903, 0.423815, 0.402958, 0.453917, 106.4554, 19.70086, 16.72001, 7.715562, 5.759398, 4.015744, 2.827112, 1.540069, 0.636755, 0.419609, 0.256458, 0.162839, 0.093635, 0.020913, -0.03394, -0.00879, -0.06103, -0.00359, 0.003105, 143.5547]
    #test_sample = [-58.14048386,	96.28108978,	-8.566257477,	14.7651329,	1.748674989,	5.317416668,	5.303733826,	3.715207815,	0.796885967,	6.523583889,	1.600593448,	1.65604043,	2.71558547,	0.096513496,	0.173773259,	2323.056352,	2515.941936,	4964.683819,	16.77641844,	12.69915723,	15.39308486,	15.90925491,	16.26104354,	16.90522976,	15.39040702,	0.459304869,	0.444025755,	0.482801884,	0.614451766,	0.484963596,	0.482585102,	0.517724693,	0.510885358,	0.587713242,	0.488834262,	0.522127807,	0.493982643,	31.06067276,	11.95944595,	7.938595295,	5.092131138,	4.087956905,	2.439917326,	1.342506528,	0.798448503,	0.586523235,	0.323691249,	0.19041954,	0.120955415,	0.083570473,	-0.02519336,	0.00363792,	0.015562645,	0.013090128,	-0.007643536,	-0.00166887,	151.9990809]
    #test_sample = [-221.895, 104.2554, -11.0523, 14.5553, 3.754919, -0.68154, 1.904632, -4.0655, -5.69395, 2.147113, -1.68033, 1.712232, 1.073156, 0.102347, 0.04686, 2156.606, 2349.564, 4341.186, 18.12844, 14.21564, 17.20104, 16.83414, 17.38763, 19.22825, 16.37985, 0.407639, 0.351877, 0.383599, 0.393179, 0.395262, 0.369378, 0.37978, 0.439627, 0.367912, 0.345032, 0.362784, 0.411323, 2.344805, 1.929811, 1.234444, 0.479714, 0.464163, 0.26029, 0.131961, 0.073802, 0.062703, 0.097076, 0.008553, 0.008487, 0.012661, 0.022418, 0.022802, 0.021475, 0.014021, 0.008022, 0.000148, 151.9991]
    #test_sample = [-232.657, 111.9534, -16.5903, 7.99917, -4.66335, -4.91768, -5.26169, -6.16442, -11.9964, -9.89747, -2.66681, -3.95212, -1.01057, 0.084925, 0.057507, 1888.532, 2102.399, 3621.173, 18.76119, 16.89827, 19.27843, 18.89106, 19.25058, 20.39699, 17.06845, 0.406623, 0.342793, 0.319599, 0.31083, 0.320032, 0.287763, 0.271514, 0.349461, 0.317948, 0.308693, 0.308151, 0.340721, 3.219592, 4.615441, 2.621299, 1.450312, 0.917039, 0.384067, 0.150892, 0.101994, 0.064968, 0.019206, 0.007738, 0.004267, 0.002132, 0.030153, 0.087395, 0.01048, 0.082316, 0.011121, 0.020261, 117.4538]
    #test_sample = [-177.368, 68.02159, -27.0147, -0.59578, -1.62976, -17.7903, -14.1234, -15.7079, -21.237, -0.89615, -7.81757, -12.014, -10.9784, 0.125276, 0.110617, 2430.098, 2323.612, 4683.158, 21.17406, 15.91632, 17.76331, 16.11456, 17.42013, 18.57351, 16.39113, 0.314214, 0.306089, 0.294186, 0.291561, 0.3161, 0.357815, 0.400759, 0.393953, 0.389299, 0.389497, 0.356083, 0.330693, 15.46079, 24.60375, 11.82809, 2.631247, 1.805652, 2.119426, 0.935718, 0.791937, 0.46623, 0.135456, 0.047647, 0.055802, 0.045321, -0.00143, -0.0021, -0.01045, 0.004858, -0.00037, 0.00339, 123.0469]
    #test_sample= [-80.22679901123047,77.39530181884766,8.31019115447998,23.20447540283203,6.752033710479736,5.72260856628418,0.9017983078956604,-1.388983130455017,0.09659327566623688,1.914087176322937,-1.8025966882705688,1.551510214805603,1.301680088043213,0.0983653149675242,0.18943744897842407,2598.3945296394413,2753.029080588071,5677.082560573739,18.976759222994783,16.130665748408507,20.472869778091862,19.205832971917225,18.73273062450443,17.59971231596049,15.982726217774635,0.3983091115951538,0.39722874760627747,0.29091235995292664,0.3427101671695709,0.27604857087135315,0.3876084089279175,0.23982864618301392,0.2695463001728058,0.3708444833755493,0.36862483620643616,0.43499088287353516,0.3407496511936188,41.82670211791992,19.128154754638672,13.952044486999512,5.670306205749512,3.812256336212158,2.8663835525512695,1.4535554647445679,0.802120566368103,0.5115773677825928,0.38186290860176086,0.17782196402549744,0.11209587007761002,0.09633588790893555,-0.049807358569182524,0.0694337306688702,-0.09005817890635033,-0.0223389492331522,0.022645385690817334,0.0028704746457239626,107.666015625]#song
    test_sample=[-211.2218017578125,94.45687103271484,13.910279273986816,24.12769889831543,11.07049560546875,7.692944526672363,3.6634228229522705,3.1084280014038086,-1.3290966749191284,3.133037567138672,0.4297395944595337,0.4010441303253174,1.7330678701400757,0.07499542053525586,0.07962702214717865,2100.99112710736,2464.681681007285,4715.766877751022,17.674741686190217,14.558223365935788,17.796702134340375,16.967835279378,17.553820535696165,18.198356414027092,17.11479991206008,0.4458516836166382,0.4211830794811249,0.43733879923820496,0.42812085151672363,0.4237234890460968,0.4102301597595215,0.4047396183013916,0.39206886291503906,0.4237532913684845,0.42188480496406555,0.4152332842350006,0.4064488410949707,12.398670196533203,3.456437349319458,1.5433555841445923,0.6639308929443359,0.5195918679237366,0.30707550048828125,0.1676693707704544,0.1295533925294876,0.07768358290195465,0.04541032388806343,0.028375037014484406,0.023168662562966347,0.01401982270181179,-0.02746183272861277,-0.042731768720675765,-0.023900075298646087,-0.002197825866573662,0.00621885324129568,-0.00552624125331964,117.45383522727273]
    try:
        # Try to load existing data
        print("Loading your dataset...")
        df = pd.read_csv('audio_features.csv')
    except FileNotFoundError:
        print("No existing dataset found. Creating sample dataset...")
        df = create_sample_data()
        # Save sample dataset
        df.to_csv('sample_audio_features.csv', index=False)
        print("Sample dataset saved as 'sample_audio_features.csv'")
    
    # Train the model
    model = train_model(df)
    
    # Make prediction on test sample
    print("\nMaking prediction on test sample...")
    prediction, probabilities = predict_audio(test_sample, model)
    
    print(f"\nPredicted class: {prediction}")
    print("\nClass probabilities:")
    for class_name, prob in probabilities.items():
        print(f"{class_name}: {prob:.2%}")

if __name__ == "__main__":
    main()