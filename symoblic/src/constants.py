TASK_INPUT_TEMPLATE = """
{task_context}

#### Now it's your turn!

**Input Grid:**
```
{query_input_grid}
```
Based on the input grid, which of the following is the correct output grid?

**Options:**

{options}

Choose the option that best represents the output grid based on the given input grid and patterns you have observed in the examples.


"""

TASK_DESCRIPTION_PATH = "../data/templates/task_description.md"
TASK_QUERY_TEMPLATE_PATH = "../data/templates/task_query.template"
TASK_QUERY_OPTION_TEMPLATE_PATH = "../data/templates/task_query_option.template"
TASK_CONTEXT_NO_OPTION_TEMPLATE_PATH = "../data/templates/task_context_no_option.template"
TASK_CONTEXT_WITH_OPTION_TEMPLATE_PATH = "../data/templates/task_context_with_option.template"

COLORS = [chr(i) for i in range(ord("A"), ord("J") + 1)]

# OPTIONS = [chr(i) for i in range(ord("a"), ord("e") + 1)]
OPTIONS = [str(i) for i in range(5)]

LOGIT_BIAS = {15: 10, 16: 10, 17: 10, 18: 10, 19: 10}
