import os

# Determine the root directory of the current script file
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))


def out_file(file_name, sentence):
    """
    Append a sentence to a text file in the project directory, printing it to console as well.

    Parameters:
        file_name (str): Name of the file (relative to ROOT_DIR) to write to.
        sentence (str): Text sentence to append to the file.
    """
    # Construct full file path
    path = os.path.join(ROOT_DIR, file_name)
    # Print the sentence to standard output
    print(sentence)
    # Open file in append mode and write the sentence with a newline
    with open(path, "a", encoding='utf8') as f:
        f.write(sentence + "\n")


def init_file(file_name):
    """
    Initialize (or clear) the given file in the project directory by writing an empty string.

    Parameters:
        file_name (str): Name of the file (relative to ROOT_DIR) to initialize.
    """
    # Construct full file path
    path = os.path.join(ROOT_DIR, file_name)
    # Open file in write mode, which clears existing content
    with open(path, 'w', encoding='utf8') as f:
        f.write("")
