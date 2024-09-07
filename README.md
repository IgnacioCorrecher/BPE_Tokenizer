# Byte Pair Encoding (BPE) Tokenizer

This project implements a Byte Pair Encoding (BPE) tokenizer, a subword tokenization algorithm commonly used in Natural Language Processing (NLP) tasks. The implementation includes both a basic version and an advanced version using regex for improved tokenization.

## Project Structure

-   `BPE_basic.py`: Basic implementation of the BPE algorithm
-   `BPE_regex.py`: Advanced implementation using regex for improved tokenization
-   `test.ipynb`: Jupyter notebook for testing and demonstrating the tokenizer
-   `data/`: Folder containing sample texts for tokenization

## File Descriptions

### BPE_basic.py

This file contains the `BytePairEncoding` class, which implements the core BPE algorithm. It includes methods for:

-   Training the tokenizer on a given text
-   Encoding text into token IDs
-   Decoding token IDs back into text

The basic version operates on UTF-8 encoded bytes of the input text.

### BPE_regex.py

This file contains the `BytePairEncodingRegex` class, which extends the basic BPE implementation with regex-based tokenization. Key features include:

-   Use of the GPT-4 split pattern for initial tokenization
-   Improved handling of subwords and special characters
-   Option to customize the regex pattern

### test.ipynb

This Jupyter notebook is used for testing and demonstrating the functionality of both BPE implementations. It likely includes:

-   Example usage of both `BytePairEncoding` and `BytePairEncodingRegex` classes
-   Comparisons between the basic and regex-based implementations
-   Visualization of tokenization results

## Data Folder

The `data/` folder contains sample texts for tokenization. These texts can be used to:

1. Train the BPE tokenizer
2. Test the encoding and decoding processes
3. Compare the performance of different tokenization strategies

To use the data:

1. Place your text files in the `data/` folder
2. Load the texts in your Python script or Jupyter notebook
3. Use the loaded texts to train and test the BPE tokenizer

For more detailed examples and usage, refer to the test.ipynb notebook.

## Results

Applying both tokenizers to the same random text, we see the following results, showcasing noticable improvements in tokenization using the regex-based approach:

`Basic BPE Tokenizer: Lenght: 9760 vs 25908 -> 2.655X compression rate`

`Regex-based BPE Tokenizer: Lenght: 9081 vs 25908 -> 2.853X compression rate`

## Medium Article

This repository is the companion code for the Medium article [Byte Pair Encoding (BPE) Tokenizer](https://medium.com/@prajwal.kumar/byte-pair-encoding-bpe-tokenizer-in-python-c6a4f7f8f3f9).
