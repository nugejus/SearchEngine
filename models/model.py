import re
import numpy as np
from collections import Counter

class Model:
    """
    A simple model for text vectorization and relevance scoring
    using FastText vectors and character bigrams.
    """

    def __init__(self):
        """
        Initialize the Model instance.
        """
        pass

    def norm(self, matrix):
        """
        Normalize each row vector in the matrix to unit length (L2 norm).

        Parameters:
            matrix (np.ndarray): 2D array where each row is a vector.

        Returns:
            np.ndarray: Row-wise normalized matrix.
        """
        return matrix / np.linalg.norm(matrix, axis=1, keepdims=True)

    def get_relevance(self, req_norm, doc_norm):
        """
        Compute relevance scores between normalized request vectors and document vectors.

        For each request vector, calculates the dot-product-based score against all document vectors,
        then sorts documents by descending score.

        Parameters:
            req_norm (np.ndarray): Array of normalized request vectors.
            doc_norm (np.ndarray): Array of normalized document vectors.

        Returns:
            list of list of tuples: Each inner list contains (doc_index, score) pairs,
            sorted by score in descending order for each request.
        """
        relevance = []
        for req in req_norm:
            # Compute scores by element-wise multiplication and summing
            scores = {i: (doc_norm[i] * req).sum() for i in range(len(doc_norm))}
            # Sort documents by score descending
            sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            relevance.append(sorted_scores)
        return relevance

    def _bigram_combination(self, iterable):
        """
        Generate consecutive character bigrams from the input iterable.

        Parameters:
            iterable (sequence): Sequence of characters or elements.

        Yields:
            sequence: Two-item slices representing a bigram.
        """
        for i in range(len(iterable) - 1):
            yield iterable[i:i+2]

    def _bigram_mean_vector(self, word, bigrams):
        """
        Approximate an out-of-vocabulary word vector by averaging its character bigram vectors.

        Parameters:
            word (str): The token to vectorize.
            bigrams (dict): Mapping from bigram string to vector (np.ndarray).

        Returns:
            np.ndarray: Averaged bigram vector of dimension 300.
        """
        # Remove non-alphanumeric characters and split into characters
        chars = list(re.sub(r'[^\w]', '', word))
        vec = np.zeros(300, dtype=np.float64)
        count = 0
        for bi in self._bigram_combination(chars):
            bi_str = ''.join(bi)
            if bi_str in bigrams:
                vec += bigrams[bi_str]
            count += 1
        # Avoid division by zero
        return vec / count if count else vec

    def _text_to_vec(self, text, fasttext_vectors, bigrams, term_idx):
        """
        Convert tokenized text into a single vector representation.

        Looks up FastText embeddings for in-vocabulary words or uses bigram-based
        approximation for out-of-vocabulary words, then averages term vectors.

        Parameters:
            text (list of str): Tokenized words of the document.
            fasttext_vectors (dict): Pre-trained word vectors.
            bigrams (dict): Pre-trained bigram vectors.
            term_idx (dict): Mapping of term to its index in the output vector.

        Returns:
            np.ndarray: Averaged 300-dimension vector for the input text.
        """
        # Initialize container for term vectors
        vecs = np.zeros((len(term_idx), 300), dtype=np.float64)
        for word in text:
            if word in fasttext_vectors:
                vecs[term_idx[word]] = fasttext_vectors[word]
            else:
                vecs[term_idx[word]] = self._bigram_mean_vector(word, bigrams)
        # Compute mean vector across all terms
        return np.mean(vecs, axis=0)

    def get_tf(self, normal_form, term_idx):
        """
        Compute the term frequency (TF) matrix for a list of tokenized documents.

        Parameters:
            normal_form (list of list of str): List of tokenized documents.
            term_idx (dict): Mapping from term to column index.

        Returns:
            np.ndarray: 2D TF matrix of shape (num_docs, num_terms).
        """
        tf = np.zeros((len(normal_form), len(term_idx)), dtype=np.float64)
        for i, doc in enumerate(normal_form):
            term_count = Counter(doc)
            for term, count in term_count.items():
                if term in term_idx:
                    tf[i, term_idx[term]] = count
        return tf
