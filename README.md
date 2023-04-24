# linguistic relatedness & transfer learning

## classification tasks

in all tasks, the input will be tokenized text.
the output label will be a string that depends on the task.

some example tasks:

- language identification (label: language code)
    * input: single sentence
    * label: language label
- script detection (label: script code)
    * input: single sentence
    * label: script label
- are these two sentences the same in meaning? (label: yes/no)
    * input: two sentences in different languages (not necessarily parallel)
    * label: yes/no
- are the languages these two sentences are in related? (label: yes/no)
    * input: two sentences in different languages (not necessarily parallel)
    * label: yes/no
