from smgvtt.semantics.text.prompts.prompt_template import prompt_for_category
if __name__=='__main__':
    for label in ['spoon','bowl','fork']:
        print(label,':',prompt_for_category(label))
