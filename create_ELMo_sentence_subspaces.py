import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import pandas as pd
import re
from gensim.parsing.preprocessing import remove_stopwords, strip_punctuation
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from datasets import load_dataset
import random
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import nltk
nltk.download('punkt')
import itertools
import os
from IPython.display import clear_output

# load_dataset("wikipedia", "20220301.en")
corpus = load_dataset("wikipedia", "20220301.simple")

elmo = hub.Module("elmo_3", trainable=True)

def get_sentences(corpus, concept):
    '''
    Gets all sentences including the exact concept.
    '''

    pattern = r'\b{}\b'.format(concept.lower())  # regex pattern for exact word match (maybe later also include concept + "'s"?)
    sentences = []
    for text in corpus['train']['text']:
        for sentence in re.split('[.!?]', text):
            if re.search(pattern, sentence.lower()):
                sentences.append([sentence])

    print(f'- {len(sentences)} sentences extracted containing the concept {concept}.')

    return sentences

# Variation 1: excluding the other concept
def get_exclude_sentences(corpus, concept, comparison_concept):
    '''
    Gets all sentences of the concept excluding the compared to concept.
    '''

    pattern = r'\b{}\b'.format(concept.lower())
    pattern2 = r'\b{}\b'.format(comparison_concept.lower())
    sentences = []
    for text in corpus['train']['text']:
        for sentence in re.split('[.!?]', text):
            if re.search(pattern, sentence.lower()):
                if re.search(pattern2, sentence.lower()):
                    continue
                else:
                    sentences.append([sentence])

    print(f'- {len(sentences)} sentences extracted containing concept A excluding concept B: {concept}-{comparison_concept}')

    return sentences

# Variation 2: including only sentences with both concepts present
def get_overlap_sentences(corpus, concept, comparison_concept):
    '''
    Gets all sentences including the exact concept and the comparison concept.
    '''
    pattern = r'\b{}\b'.format(concept.lower())
    pattern2 = r'\b{}\b'.format(comparison_concept.lower())
    sentences = []
    for text in corpus['train']['text']:
        for sentence in re.split('[.!?]', text):
            if re.search(pattern, sentence.lower()):
                if re.search(pattern2, sentence.lower()):
                    sentences.append([sentence])
    
    print(f'- {len(sentences)} sentences extracted containing the combination of concepts: {concept}-{comparison_concept}')

    return sentences

def preprocess_sentences(sentences_list, remove_stopword=False):
    
    # Make all words lowercase
    sentences = [[word.lower() for word in sentence] for sentence in sentences_list]

    # Remove stopwords
    if remove_stopword:
        sentences = [[remove_stopwords(word) for word in sentence] for sentence in sentences]

    # Strip the punctuation of the words
    sentences = [[strip_punctuation(word) for word in sentence] for sentence in sentences]

    # Remove numbers
    # sentences = [[word for word in sentence if not word.isdigit()] for sentence in sentences]

    # Lemmatize words
    sentences = [[lemmatizer.lemmatize(word) for word in sentence] for sentence in sentences]

    # Remove sublists to put all sentences in one big list
    sentences = [item for sublist in sentences for item in sublist]

    # Remove any extra white spaces
    sentences = [' '.join(sentence.split()) for sentence in sentences]

    return sentences

def generate_elmo_embeddings(sentences, concept, layer_to_use, batch_size):
        '''
        Creates embeddings for all sentences.
        '''

        with tf.compat.v1.Session() as sess:
            sess.run(tf.compat.v1.global_variables_initializer())
            sess.run(tf.compat.v1.tables_initializer())
            
            concept_embedding = []
            for i in range(0, len(sentences), batch_size):
                print(f'Batch of {batch_size}')
                batch_sentences = sentences[i:i+batch_size]
                embeddings = elmo(batch_sentences, signature="default", as_dict=True)[layer_to_use]
                batch_embeddings = embeddings.eval()
                batch_embeddings = np.mean(batch_embeddings, axis=1)
                concept_embedding.append(batch_embeddings)

            concept_embeddings = np.concatenate(concept_embedding, axis=0)

        return pd.DataFrame(concept_embeddings, columns=[f"dim{i}" for i in range(1024)])

def horn_parallel_analysis(data, num_iterations=100, scale=True):
    '''
    Perform Horn's Parallel Analysis to get number of components.
    '''

    if scale:
        scaler = StandardScaler(with_mean=True, with_std=True)
        data = scaler.fit_transform(data)

    n, p = data.shape

    # Generate random data with the same dimensions as the input data
    rand_data = np.random.normal(loc=np.mean(data), scale=np.std(data), size=(n, p, num_iterations))

    # Compute the eigenvalues for both the input data and the random data
    pca = PCA(n_components=p)
    pca.fit(data)
    observed_eigenvals = pca.explained_variance_

    rand_eigenvals = np.zeros((p, num_iterations))
    for i in range(num_iterations):
        pca.fit(rand_data[:,:,i])
        rand_eigenvals[:,i] = pca.explained_variance_

    # Compute the 95th percentile of the random eigenvalues
    percentile = np.percentile(rand_eigenvals, 95, axis=1)

    # Determine the number of principal components to keep
    num_components = np.sum(observed_eigenvals > percentile)

    print("Number of principal components to retain:", num_components)

    plot_data = pd.DataFrame({'observed_eigenvals':observed_eigenvals,'percentile':percentile})

    return num_components, plot_data

def create_subspace(sentences_list, concept, sentences_to_include, elmo_layer_to_use, remove_stopword, batch_size):
    
    print('Getting sentences...')
    if len(sentences_list)>=sentences_to_include:
        random.seed(0)
        init_len = len(sentences_list)
        sentences_list = [sentence for sentence in sentences_list if len(sentence)<50]
        print(init_len - len(sentences_list),'sentences removed with over 50 words.')
        sentences_list = random.sample(sentences_list, sentences_to_include)
    else:
        print(f'WARNING NOT ENOUGH SENTENCES!')
    print(f'    - {len(sentences_list)} of these sampled.')
    
    print('Preprocessing sentences...')
    sentences = preprocess_sentences(sentences_list, remove_stopword)

    print('Creating embeddings...')
    embeddings = generate_elmo_embeddings(sentences, concept, elmo_layer_to_use, batch_size)

    print('Determining number of components...')
    num_components, plot_data = horn_parallel_analysis(embeddings.T)

    print('Extracting principal components...')
    pca = PCA(n_components=num_components).fit(embeddings.T)
    principal_components = pca.transform(embeddings.T)

    pca_df = pd.DataFrame(principal_components, columns=[f'PC_{i}' for i in range(1, num_components+1)])
    pca_df = pca_df.T
    pca_df.columns = [f'dim{i}' for i in range(len(pca_df.columns))]    

    print('Done :)')
    
    return pca_df

sentences = {
    'innovation': get_sentences(corpus,'innovation'),
    'invention': get_sentences(corpus,'invention'),
    'renovation': get_sentences(corpus,'renovation'),
    'technology': get_sentences(corpus,'technology'),
    'creativity': get_sentences(corpus,'creativity'),
    'implementation': get_sentences(corpus,'implementation'),
}

# Variations
sentences_var = list(sentences.keys())
remove_stopword = [False, True]
elmo_layers = ['elmo','lstm_outputs1','lstm_outputs2']

combinations = list(itertools.product(sentences_var, remove_stopword, elmo_layers))
print(f'Creating {len(combinations)} different subspaces.')


for j,i in enumerate(combinations):
    print('Iteration:',j)

    file = f"new_subspaces_sentence_level/{i[0]}&{str(i[1])}&{i[2]}.csv"

    if os.path.isfile(file):
        continue
    
    else:
        subspace = create_subspace(
            sentences_list=sentences[i[0]],
            concept=i[0].split('-')[0],
            sentences_to_include=100,
            elmo_layer_to_use=i[2],
            remove_stopword=i[1],
            batch_size=25
            )
        subspace.to_csv(file)

    clear_output()