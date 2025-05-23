from model import Model

class Fasttext(Model):
    """
    FastText-based relevance model inheriting from the base Model class.

    Uses pre-trained FastText embeddings (or bigram approximations) to
    convert text into vectors and computes relevance via cosine similarity.
    """
    def __init__(self):
        """
        Initialize the Fasttext instance by calling the superclass constructor.
        """
        super().__init__()

    def __call__(
        self,
        normalized_requests,
        normalized_documents,
        fasttext_vectors,
        bigrams,
        term_idx
    ):
        """
        Compute relevance rankings between requests and documents using FastText embeddings.

        Steps:
            1. Convert each request and document into a vector representation
               using _text_to_vec (FastText lookup or bigram fallback).
            2. Normalize all vectors to unit length.
            3. Compute relevance scores based on cosine similarity.

        Parameters:
            normalized_requests (list of list of str): Tokenized request texts.
            normalized_documents (list of list of str): Tokenized document texts.
            fasttext_vectors (dict): Mapping from word to its FastText embedding.
            bigrams (dict): Mapping from character bigram to vector for OOV words.
            term_idx (dict): Mapping from term to index position (used by _text_to_vec).

        Returns:
            list of list of tuples: For each request, a sorted list of (doc_index, score)
            in descending order of relevance.
        """
        # Build vector representations for each request
        request_vectors = []
        for request in normalized_requests:
            vec = self._text_to_vec(request, fasttext_vectors, bigrams, term_idx)
            request_vectors.append(vec)

        # Build vector representations for each document
        document_vectors = []
        for document in normalized_documents:
            vec = self._text_to_vec(document, fasttext_vectors, bigrams, term_idx)
            document_vectors.append(vec)

        # Normalize vectors and compute relevance scores
        # Note: get_relevance expects (req_norm, doc_norm), so swap order
        req_norms = self.norm(np.array(request_vectors))
        doc_norms = self.norm(np.array(document_vectors))
        return self.get_relevance(req_norms, doc_norms)
