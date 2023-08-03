import pandas as pd

def icl_setup(stage, context, document, principles, prev_best_response, icl_examples=None):
    """ Helper for obtaining the prompts: from run_algo_<platform>, obtain the prompt according to the current stage of the iterative improvement process.
    """
    if stage == 'answerable':
        icl_examples = 'data/icl_samples/icl_answerability.jsonl'
        icl_prompt = icl_answerable(icl_examples, context, document)
    elif stage == 'initial generation':
        icl_prompt = icl_init_gen(icl_examples, context, document)
    elif stage == 'user query':
        icl_examples = 'data/icl_samples/icl_query_gen.jsonl'
        icl_prompt = icl_user_query(icl_examples, context, document)
    elif stage == 'iterative improvement':
        icl_examples = {}
        for principle in principles:
            icl_examples[principle] = 'data/icl_samples/iterate_icl_manual_' + principle + '_original3.jsonl'
        icl_prompt = icl_iter_improve(icl_examples, context, document, principles, prev_best_response)
    elif stage == 'iterative improvement multi-turn':
        icl_examples = {}
        for principle in principles:
            icl_examples[principle] = 'data/icl_samples/iterate_icl_manual_' + principle + '_original3.jsonl'
        icl_prompt = icl_iter_improve_multi_turn(icl_examples, context, document, principles, prev_best_response)
    elif stage == 'iterative improvement zero shot':
        icl_prompt = icl_iterate_zero_shot(context, document, principles, prev_best_response)
    return icl_prompt

def icl_answerable(icl_examples, context, document):
    """ Determine whether or not the query (in the context) is answerable, given the document
    """

    instruction = (
        "Provided is a dialog between two speakers, User and Agent. Determine whether the context is answerable, given the document. If it is answerable, return 'positive'. If it is not answerable, return 'negative'."
    )

    icl_examples = pd.read_json(icl_examples, orient='records', lines=True)
    icl_sample_template = """
document: {document}

context: {context}
label: {answerable}"""

    formatted_samples = []
    for _, example in icl_examples.iterrows():
        single_sample = icl_sample_template.format(document=example['document'], context=example['context'], answerable=example['label'])
        formatted_samples.append(single_sample)
    
    joint_instruction = instruction + '\n\n###\n\n'.join(formatted_samples) + '\n\n###\n\n'
    label = 'label: '
    full_query = f'{joint_instruction}document: {document}\n\ncontext: {context}\n{label}'
    return full_query

def icl_init_gen(icl_examples, context, document):
    """ Get the instruction and prompt for initial response generation, using and including the in-context examples of initial responses
    """

    instruction = (
        "Provided is a dialog between two speakers, User and Agent. Generate a response that is coherent with the dialog history and the provided document. Desired traits for responses are: 1) Specific - The response contains specific content, and 2) Accurate - The response is correct and factual with respect to the document."
    )
    icl_examples = pd.read_json(icl_examples, orient="records", lines=True)
    icl_sample_template = """
document: {document}

context: {context}
Agent: {response}"""
    formatted_samples = []
    for _, example in icl_examples.iterrows():
        single_sample = icl_sample_template.format(document=example['document'], context=example['context'], response=example['response'])
        formatted_samples.append(single_sample)

    joint_instruction = instruction + '\n\n###\n\n'.join(formatted_samples) + '\n\n###\n\n'
    perspective = 'Agent: '
    full_query = f'{joint_instruction}\n\ndocument: {document}\n\ncontext: {context}\n{perspective}'
    return full_query


def icl_user_query(icl_examples, context, document):
    """ Get prompt for user query generation using in-context examples of a natural dialogue and user queries related to the document
    """

    instruction = (
        "Provided is a dialog between two speakers, User and Agent. Generate a new question, posed by the user, that is coherent with the dialog history and contains specfic content."
    )
    icl_examples = pd.read_json(icl_examples, orient="records", lines=True)

    ### here, the context ends off on a user turn (a new user query)
    icl_sample_template = """
    document: {document}
    
    context: {context}"""

    formatted_samples = []
    for _, example in icl_examples.iterrows():
        single_sample = icl_sample_template.format(document=example['document'], context=example['context'])
        formatted_samples.append(single_sample)

    joint_instruction = instruction + '\n\n###\n\n'.join(formatted_samples) + '\n\n###\n\n'
    ### the context for the current dialogue ends off on an Agent response, so the next turn is a user query
    perspective = 'User: '
    full_query = f'{joint_instruction}\n\ndocument: {document}\n\ncontext:{context}\n{perspective}'
    return full_query


def icl_iter_improve(icl_examples, context, document, principles, prev_best_response):
    """ Get the instruction and prompt for iterative improvement for each principle, using the in-context examples of improvement for each principle
    and including them in the prompt as demonstrations. 
    """

    principle_queries = []
    for principle in principles:
        icl_examples = pd.read_json(icl_examples[principle], orient="records", lines=True)
        instruction = (
            """We want to improve the previous response to make it more {principle}. To aid in this process, we provide examples of incremental improvement on {principle_type}, where Agent response 2 is more {principle} than Agent response 1."""
        )
        if principle == 'specific':
            instruction = instruction.format(principle=principle, principle_type='specificity')
        elif principle == 'accurate':
            instruction = instruction.format(principle=principle, principle_type='accuracy')

        icl_improvement_template = """
document: {document}
        
context: {context}
Agent response 1 (not {principle}): {less_principle_response}
        
Let's make this response more {principle}. 
        
Agent response 2 (more {principle}): {more_principle_response}"""
        formatted_samples = []
        for _, example in icl_examples.iterrows():
            example = example.to_dict()
            if principle == 'specific':
                icl_sample = icl_improvement_template.format(principle=principle, document=example['document'], context=example['context'], 
                                                         less_principle_response=example['less_specific_response'], more_principle_response=example['more_specific_response'])
                conv_improvement_format = f"""
document: {document}

context: {context}
Agent response 1 (not specific): {prev_best_response}

Let's make this response more specific.

Agent response 2 (more specific): 
"""
            elif principle == 'accurate':
                icl_sample = icl_improvement_template.format(principle=principle, document=example['document'], context=example['context'],
                                                             less_principle_response=example['less_accurate_response'], more_principle_response=example['more_principle_response'])
                conv_improvement_format = f"""
document: {document}

context: {context}
Agent response 1 (not accurate): {prev_best_response}

Let's make this response more accurate.

Agent response 2 (more accurate): 
"""
            icl_sample = (instruction + icl_sample).strip()
            formatted_samples.append(icl_sample)
        principle_icl_prompt = '\n\n###\n\n'.join(formatted_samples) + '\n\n###\n\n'

        full_prompt = f"{principle_icl_prompt}{conv_improvement_format}"

        principle_queries.append(full_prompt)
    #print(principle_queries)
    return principle_queries


def icl_iter_improve_multi_turn(icl_examples, context, document, principles, prev_best_response):
    """ Get the instruction and prompt for iterative improvement for each principle, using the in-context examples of improvement for each principle
    and including them in the prompt as demonstrations. 
    """

    principle_queries = []
    for principle in principles:
        icl_examples = pd.read_json(icl_examples[principle], orient="records", lines=True)
        instruction = (
            """We want to improve the previous response to make it more {principle}. To aid in this process, we provide examples of incremental improvement on {principle_type}, where Agent response 2 is more {principle} than Agent response 1."""
        )
        if principle == 'specific':
            instruction = instruction.format(principle=principle, principle_type='specificity')
        elif principle == 'accurate':
            instruction = instruction.format(principle=principle, principle_type='accuracy')

        icl_improvement_template = """
document: {document}
        
context: {context}
Agent response 1 (not {principle}): {less_principle_response}
        
User: Please make your response more {principle}. 
        
Agent response 2 (more {principle}): {more_principle_response}"""
        formatted_samples = []
        for _, example in icl_examples.iterrows():
            example = example.to_dict()
            if principle == 'specific':
                icl_sample = icl_improvement_template.format(principle=principle, document=example['document'], context=example['context'], 
                                                         less_principle_response=example['less_specific_response'], more_principle_response=example['more_specific_response'])
                conv_improvement_format = f"""
document: {document}

context: {context}
Agent response 1 (not specific): {prev_best_response}

User: Please make this response more specific.

Agent response 2 (more specific): """
            elif principle == 'accurate':
                icl_sample = icl_improvement_template.format(principle=principle, document=example['document'], context=example['context'],
                                                             less_principle_response=example['less_accurate_response'], more_principle_response=example['more_principle_response'])
                conv_improvement_format = f"""
document: {document}

context: {context}
Agent response 1 (not accurate): {prev_best_response}

User: Please make this response more accurate.

Agent response 2 (more accurate): """
            icl_sample = (instruction + icl_sample).strip()
            formatted_samples.append(icl_sample)
        principle_icl_prompt = '\n\n###\n\n'.join(formatted_samples) + '\n\n###\n\n'

        full_prompt = f"{principle_icl_prompt}{conv_improvement_format}"

        principle_queries.append(full_prompt)
    return principle_queries


def icl_iterate_zero_shot(context, document, principles, prev_best_response):
    """ Get the instruction and prompt for iterative improvement of the response, without any in-context examples of improvement. 
    """

    principle_queries = []
    for principle in principles:
        instruction = (
            """We want to improve the previous response to make it more {principle}. To aid in this process, we provide examples of incremental improvement on {principle_type}, where Agent response 2 is more {principle} than Agent response 1."""
        )
        if principle == 'specific':
            instruction = instruction.format(principle=principle, principle_type='specificity')
            conv_improvement_format = f"""
document: {document}

context: {context}
Agent response 1 (not specific): {prev_best_response}

Let's make this response more specific.

Agent response 2 (more specific): 
"""
        elif principle == 'accurate':
            instruction = instruction.format(principle=principle, principle_type='accuracy')
            conv_improvement_format = f"""
document: {document}

context: {context}
Agent response 1 (not specific): {prev_best_response}

Let's make this response more specific.

Agent response 2 (more specific): 
"""
        full_prompt = instruction.strip() + '\n\n' + conv_improvement_format
        principle_queries.append(full_prompt)
    return principle_queries


def comparison_prompt(dimension, outputs, context, document, prev_best_response, principle_count=None, principle=None):
    """ Prompting a model to choose which output is better on a particular principle.
    This can be used for selection of the best output across multiple principles' greedy improvements, or determining if the new response
    constitutes an improvement over the previous response on that principle, as an stopping criterion for the refinement process. 
    
    Note: Currently not being used, but could possibly be applied for model-based evaluation between two responses. 
    """

    if dimension == 'general':
        # Here, we're comparing the outputs from the different principles
        prompt_base = f"""
document: {document}

context: {context}
Agent:

Here are {principle_count} agent responses:
"""
        prompt_responses = ''
        for i in range(principle_count):
            prompt_responses += f"""
Response {i}: {outputs[i]}
""" 
        instruction = (
            """Which response is best, given the context and the document?"""
        )
        prompt = prompt_base + prompt_responses + instruction

    elif dimension == 'principle':
        # Here, we are comparing the previous best with the current answer, on a particular principle
        prompt_base = prompt_base = f"""
document: {document}

context: {context}
Agent:

Here are the two agent responses:
"""
        prompt_responses = f"""
Response 1: {prev_best_response}
Response 2: {outputs[0]}
"""
        instruction = f"""Which response is more {principle}, given the context and the document: Response 1 or Response 2?"""
        prompt = prompt_base + prompt_responses + instruction

    return prompt
