import numpy as np
from model import Model

class Tf_Idf(Model):
    """
    Term Frequency-Inverse Document Frequency (TF-IDF) relevance model
    inheriting from the base Model class.
    """
    def __init__(self):
        """
        Initialize the Tf_Idf instance by calling the superclass constructor.
        """
        super().__init__()

    def __call__(self, normal_form, normal_request, term_idx):
        """
        Compute relevance rankings between documents and requests using TF-IDF.

        Steps:
            1. Calculate TF matrices for documents and requests.
            2. Compute document frequency (DF) for each term.
            3. Calculate inverse document frequency (IDF) using DF.
            4. Weight TF matrices by IDF to obtain TF-IDF representations.
            5. Normalize TF-IDF vectors to unit length.
            6. Compute relevance scores via cosine similarity.

        Parameters:
            normal_form (list of list of str): Tokenized documents.
            normal_request (list of list of str): Tokenized query texts.
            term_idx (dict): Mapping of term to its index in the vector.

        Returns:
            list of list of tuples: For each request, a sorted list of (doc_index, score)
            in descending order of relevance.
        """
        # Generate term-frequency matrices
        document_tf = self.get_tf(normal_form, term_idx)
        request_tf = self.get_tf(normal_request, term_idx)

        # Compute document frequency: count of documents where term appears
        df = np.sum(document_tf > 0, axis=0)
        # Compute inverse document frequency with log-scaling
        idf = np.log10(len(normal_form) / df)

        # Weight term-frequency by IDF for documents and requests
        tf_idf_document = document_tf * idf
        tf_idf_request = request_tf * idf

        # Normalize TF-IDF vectors to unit length
        doc_norms = self.norm(tf_idf_document)
        req_norms = self.norm(tf_idf_request)

        # Return relevance rankings based on normalized TF-IDF
        return self.get_relevance(req_norms, doc_norms)
