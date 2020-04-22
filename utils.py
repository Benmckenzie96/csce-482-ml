def txt_to_list(words_loc):
    with open(words_loc, 'rb') as f:
        words = [line.strip() for line in f]
    return words
