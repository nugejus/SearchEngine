from sentence_transformers import SentenceTransformer
from model import Model

class LaBSE(Model):
    """
    Semantic relevance model using Facebook's LaBSE transformer.

    Inherits from the base Model class and computes cosine similarities
    between sentence embeddings of requests and documents.
    """
    def __init__(self):
        """
        Initialize the LaBSE instance by calling the superclass constructor.
        No additional setup is required.
        """
        super().__init__()

    def __call__(self, initial_requests, initial_documents):
        """
        Compute relevance rankings using the 'sentence-transformers/LaBSE' model.

        Steps:
            1. Load the pretrained LaBSE SentenceTransformer model.
            2. Encode requests and documents into dense vectors,
               performing unit-length normalization internally.
            3. Apply L2 normalization to embeddings to ensure consistency.
            4. Compute relevance scores via cosine similarity between
               request and document embeddings.

        Parameters:
            initial_requests (list of str): Raw request or query sentences.
            initial_documents (list of str): Raw document sentences or passages.

        Returns:
            list of list of tuples: For each request, returns a sorted list of
            (document_index, score) pairs in descending relevance order.
        """
        # Load the LaBSE SentenceTransformer model
        model = SentenceTransformer('sentence-transformers/LaBSE')

        # Encode the request sentences; normalize_embeddings=True yields unit vectors
        request_embeddings = model.encode(
            initial_requests,
            normalize_embeddings=True
        )
        # Encode the document sentences similarly
        document_embeddings = model.encode(
            initial_documents,
            normalize_embeddings=True
        )

        # Further ensure embeddings are L2-normalized
        req_norm = self.norm(request_embeddings)   # shape: (num_requests, dim)
        doc_norm = self.norm(document_embeddings) # shape: (num_documents, dim)

        # Return relevance rankings between normalized request and document embeddings
        return self.get_relevance(req_norm, doc_norm)
