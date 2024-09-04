from constants import (
    COLORS,
    TASK_CONTEXT_NO_OPTION_TEMPLATE_PATH,
    TASK_CONTEXT_WITH_OPTION_TEMPLATE_PATH,
    TASK_DESCRIPTION_PATH,
    TASK_QUERY_OPTION_TEMPLATE_PATH,
    TASK_QUERY_TEMPLATE_PATH,
)

with open(TASK_DESCRIPTION_PATH) as f:
    TASK_DESCRIPTION = f.read()

with open(TASK_CONTEXT_NO_OPTION_TEMPLATE_PATH) as f:
    TASK_CONTEXT_NO_OPTION_TEMPLATE = f.read()


with open(TASK_CONTEXT_WITH_OPTION_TEMPLATE_PATH) as f:
    TASK_CONTEXT_WITH_OPTION_TEMPLATE = f.read()

with open(TASK_QUERY_TEMPLATE_PATH) as f:
    TASK_QUERY_TEMPLATE = f.read()

with open(TASK_QUERY_OPTION_TEMPLATE_PATH) as f:
    TASK_QUERY_OPTION_TEMPLATE = f.read()


def convert_grid_to_text(grid):
    return "\n".join([" ".join([COLORS[cell] for cell in row]) for row in grid])


def generate_prompt(input_data, with_options=False):
    prompts = {}
    context = ""
    prompts["system"] = TASK_DESCRIPTION
    prompts["context"] = []
    prompts["query"] = []
    TASK_CONTEXT_TEMPLATE = (
        TASK_CONTEXT_WITH_OPTION_TEMPLATE if with_options else TASK_CONTEXT_NO_OPTION_TEMPLATE
    )
    for i, data_i in enumerate(input_data):
        x, y, options, label = data_i
        x_t, y_t = convert_grid_to_text(x), convert_grid_to_text(y)
        options_t = [convert_grid_to_text(option) for option in options]
        prompts["context"].append(context)
        query_options = [
            TASK_QUERY_OPTION_TEMPLATE.format(i=i, option=option) for i, option in enumerate(options_t)
        ]
        query_options_message = "\n".join(query_options)
        query = TASK_QUERY_TEMPLATE.format(input_grid=x_t, options=query_options_message)
        prompts["query"].append(query)
        context = TASK_CONTEXT_TEMPLATE.format(
            i=i, input_grid=x_t, output_grid=y_t, options=query_options_message, label=label
        )
    return prompts


def generate_prompt_v2(input_data, n_samples, n_queries, with_options=False):
    prompts = {}
    context = ""
    prompts["system"] = TASK_DESCRIPTION
    prompts["context"] = []
    prompts["query"] = []
    TASK_CONTEXT_TEMPLATE = (
        TASK_CONTEXT_WITH_OPTION_TEMPLATE if with_options else TASK_CONTEXT_NO_OPTION_TEMPLATE
    )
    for i in range(n_samples):
        data_i = input_data[i]
        x, y, options, label = data_i
        x_t, y_t = convert_grid_to_text(x), convert_grid_to_text(y)
        options = [convert_grid_to_text(option) for option in options]
        context_options = [
            TASK_QUERY_OPTION_TEMPLATE.format(i=i, option=option) for i, option in enumerate(options)
        ]
        context_options_message = "\n".join(context_options)
        context = TASK_CONTEXT_TEMPLATE.format(
            i=i, input_grid=x_t, output_grid=y_t, options=context_options_message, label=label
        )
        prompts["context"].append(context)
    for j in range(n_queries):
        query_j = input_data[n_samples + j]
        x_q, y_q, options_q, label_q = query_j
        options_q = [convert_grid_to_text(option) for option in options_q]
        x_q, y_q = convert_grid_to_text(x_q), convert_grid_to_text(y_q)
        query_options = [
            TASK_QUERY_OPTION_TEMPLATE.format(i=o, option=option) for o, option in enumerate(options_q)
        ]
        query_options_message = "\n".join(query_options)
        context_q = TASK_CONTEXT_TEMPLATE.format(
            i=i, input_grid=x_q, output_grid=y_q, options=context_options_message, label=label
        )
        query = TASK_QUERY_TEMPLATE.format(input_grid=x_q, options=query_options_message)
        prompts["query"].append(query)
    return prompts
