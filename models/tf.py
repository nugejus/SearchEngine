from model import Model

class Tf(Model):
    """
    Term Frequency (TF) relevance model inheriting from the base Model class.
    """
    def __init__(self):
        """
        Initialize the Tf instance by calling the superclass constructor.
        """
        super().__init__()

    def __call__(self, normal_form, normal_request, term_idx):
        """
        Compute relevance rankings between documents and requests using term frequency.

        Steps:
            1. Calculate TF matrices for documents and requests.
            2. Normalize these matrices to unit length.
            3. Compute relevance scores via cosine similarity.

        Parameters:
            normal_form (list of list of str): Tokenized documents.
            normal_request (list of list of str): Tokenized query or request texts.
            term_idx (dict): Mapping of term to its index in the TF vector.

        Returns:
            list of list of tuples: For each request, a sorted list of (doc_index, score)
            in descending order of relevance.
        """
        # Generate term-frequency matrix for documents
        document_tf = self.get_tf(normal_form, term_idx)
        # Generate term-frequency matrix for requests
        request_tf = self.get_tf(normal_request, term_idx)

        # Normalize each row vector to unit length
        doc_norms = self.norm(document_tf)
        req_norms = self.norm(request_tf)

        # Return relevance rankings based on normalized TF vectors
        return self.get_relevance(req_norms, doc_norms)
