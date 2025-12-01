from langgraph.graph.state import StateGraph, END
from langchain_core.runnables.base import RunnableLambda
import dotenv
from langchain_core.tools import tool
from typing import Annotated, TypedDict, List
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
import os
import operator

dotenv.load_dotenv()
llm = ChatOpenAI(model = 'gpt-5-nano')

class JudgeAnswer(BaseModel):

    '''Call this tool when you are ready to submit the final task assesment'''
    result: bool = Field(description='the result of the answer assesment, can be only True or False') # TypeScript :(((
    just: str = Field(description='the justification of the result. The the judge resturned that result?')

class QuestionHintDict(TypedDict):
    question: str
    hint: str

class QuizzState(TypedDict):

    hitPoints: int
    score: int
    tokens_so_far: float
    topic: str
    difficulty: str
    user_answers: Annotated[List[BaseMessage], add_messages]
    questions: Annotated[List[BaseMessage], add_messages]
    assessments: Annotated[List[BaseMessage], add_messages]
    hints: Annotated[List[QuestionHintDict], operator.add]
    need_to_score: int

def system_user_prompt(
    system_prompt: str,
    user_prompt: str
):
    system_template = SystemMessagePromptTemplate.from_template(system_prompt)
    user_template = HumanMessagePromptTemplate.from_template(user_prompt)

    final_prompt = ChatPromptTemplate.from_messages([
        system_template,
        user_template
    ])

    return final_prompt

def create_prompt(
    model: ChatOpenAI,
    output_key: str,
    prompt: ChatPromptTemplate
):
    def _func(input: dict):

        prompts = prompt.invoke(input).to_messages()
        response = model.invoke(prompts)

        if isinstance(response, BaseMessage):
            raw_text = response.content.strip()
        else:
            raw_text = response

        has_tool_calls = hasattr(response, 'tool_calls') and bool(response.tool_calls)
        
        if not raw_text and not has_tool_calls:
            raise ValueError(f"create_prompt function error: {prompts}")

        tokens_used = 0
        if hasattr(response, "usage_metadata") and response.usage_metadata:
            tokens_used = response.usage_metadata.get("total_tokens", 0)

        elif hasattr(response, "response_metadata"):
             token_usage = response.response_metadata.get("token_usage", {})
             tokens_used = token_usage.get("total_tokens", 0)

        prev_tokens = input.get("tokens_so_far", 0)
        new_total_tokens = tokens_used + prev_tokens

        return {
            **input,
            output_key: raw_text,
            "api_response":response, 
            "tokens_so_far": new_total_tokens,
            "prompt_messages": prompts
        }

    return RunnableLambda(_func)

@tool
def get_hint_tool(question: str, topic: str):
    '''
    A tool to receive a hint when the user asks for it. To call this tool 
    user MUST ask you for a help or hint DIRECTLY!! The hint cannot be called 
    as a third option. 
    '''

    prompt = system_user_prompt(
        system_prompt='''
        You are a helpful assistant that provides hits to the question mentioned in context.
        You never answer directly and indirectly to the question provided. 
        The hint should be obvious to the user.
        The hint should be 1-2 sentences and only this. 
        You should return a normal string with the hint without any additional information. 
        The question is related to the {topic}
        You cannot provide any direct and indirect answer to the question. It needs to be a hint. 
        ''',
        user_prompt='''
        The question:
        {question}
        '''
    )
    prompt = create_prompt(llm, 'resp', prompt)
    res = prompt.invoke({
        'question': question,
        'topic': topic
    })

    return {
        'hint': str(res.get('resp')).strip(),
        'api_response': res.get('api_response')
    } 
    
tools = [get_hint_tool, JudgeAnswer]

def stats_node(state: QuizzState):
    
    os.system('clear all')

    topic = state['topic']
    tokens_so_far = state['tokens_so_far']

    current_score = int(state['score'])
    current_hitpoints = int(state['hitPoints'])

    print(f'Your current score: {current_score}')
    print(10*'-')
    print(f'Your current hit points: {current_hitpoints}')
    print(10*'-')
    print(f'Topic: {topic}')
    print(10*'-')
    print(f'Tokens so far: {tokens_so_far}')

def calc_tokens(result, state: QuizzState):

    tokens_so_far = state['tokens_so_far']
    tokens_used = str(result.get("tokens_so_far", 0))
    new_token_so_far = int(tokens_so_far) + int(tokens_used)

    return new_token_so_far

def quiz_generator_node(state: QuizzState):

    prompt = system_user_prompt(
        system_prompt='''
            You are a dynamic Quiz Master. 
            Current Topic: {topic}
            Difficulty: {difficulty}
            
            Rules:
            1. Ask exactly one multiple-choice question.
            2. Provide 4 options (A, B, C, D).
            3. Do NOT reveal the answer yet.
            4. User will provide the answer manually
            5. Do not ask the questions that were already asked: here is the list: {questions}
            6. Your questions should be different type than the questions already asked. Please see the questions above (already asked)

        ''', 
        user_prompt='''
            Generate question according to system prompt.
        ''' 
    )
    
    topic = state['topic']
    difficulty = state['difficulty']
    already_asked_questions_aimessage = state['questions']
    already_asked_questions_content = [aimessage_question.content for aimessage_question in already_asked_questions_aimessage]

    prompt = create_prompt(llm,'resp', prompt)
    
    result = prompt.invoke({
        "topic": topic,
        "difficulty": difficulty,
        "questions": already_asked_questions_content
    })

    question_to_print = str(result.get("resp")).strip()
    question_as_aimessage = result.get("api_response")
    
    new_token_so_far = calc_tokens(result, state) # Tokens can be counted at the end of Agent job or at reducer function

    print(10*'-')
    print(question_to_print)

    return {
        'questions': [question_as_aimessage],
        'tokens_so_far': new_token_so_far
    }

def human_node(state:TypedDict):
    pass

def hint_node(state: QuizzState):

    last_assessment = state['assessments'][-1]
    
    if not last_assessment.tool_calls:
        return {"hints": []}

    tool_call_data = last_assessment.tool_calls[0]
    function_args = tool_call_data['args']

    generated_hint_text = get_hint_tool.invoke(function_args)

    question_hints: QuestionHintDict = {
        'question': function_args['question'],
        'hint': generated_hint_text['hint']
    }

    new_token_so_far = calc_tokens(generated_hint_text, state)
    print(10*'-')
    print(f'Hint: {generated_hint_text['hint']}')

    return{
        'hints': [question_hints],
        'tokens_so_far': new_token_so_far
    }

def judge_node(state: QuizzState):

    llm_with_tools = llm.bind_tools(tools)

    prompt = system_user_prompt(
        system_prompt='''

        You are a quiz judge.
        - If the user asks for help or is stuck, call the 'provide_hint' tool.
        - If the user provides an answer, YOU MUST call the 'JudgeAnswer' tool to submit your verdict.

        ''',
        user_prompt='''
        Question:
        {question}
        User asnwer:
        {user_answer}
        '''
    )
    topic = state['topic']
    question = state['questions'][-1].content
    user_answer = state['user_answers'][-1].content

    prompt = create_prompt(llm_with_tools,'resp', prompt)
    result = prompt.invoke({
        "topic": topic,
        "user_answer": user_answer,
        "question":question
    })

    new_token_so_far = calc_tokens(result, state)
    assesment = result.get("api_response")

    return {
        'assessments': [assesment],
        'tokens_so_far': new_token_so_far
    }

def parser_node(state: TypedDict):

    last_assesment = state['assessments'][-1]
    parser_call = last_assesment.tool_calls[0]
    assessment_result = JudgeAnswer(**parser_call['args'])

    print(10*'-')
    print(f'Result: {assessment_result.result}')
    print(f'Justification: {assessment_result.just}')

    current_score = int(state['score'])
    current_hitpoints = int(state['hitPoints'])

    if assessment_result.result:
        current_score += 1
    else:
        current_hitpoints -= 1

    return {
        'score': current_score,
        'hitPoints': current_hitpoints
    }

def after_assesment_router(state: TypedDict):

    last_assesment = state['assessments'][-1]

    if not last_assesment.tool_calls:
        print("An Error occurred")
        return "__end__"

    tool_name = last_assesment.tool_calls[0]['name']

    if tool_name == 'get_hint_tool':
        return 'hint_tool'
    elif tool_name == 'JudgeAnswer':
        return 'parser'

    print("An Error occurred")
    return '__end__'

def final_router(state: TypedDict):

    current_score = int(state['score'])
    current_hitpoints = int(state['hitPoints'])
    need_to_score = int(state['need_to_score'])

    if current_hitpoints == 0:
        print('You lost')
        return '__end__'

    elif current_score == 3:
        print('You won')
        return '__end__'
    
    elif current_score < need_to_score:
        return 'generator'
    
    else:
        print("An Error occurred")
        return '__end__'

     
workflow = StateGraph(QuizzState)
workflow.set_entry_point('stats')

workflow.add_node('generator', quiz_generator_node)
workflow.add_node('human_answer', human_node)
workflow.add_node('human_next', human_node)
workflow.add_node('judge', judge_node)
workflow.add_node('hint_tool', hint_node)
workflow.add_node('parser', parser_node)
workflow.add_node('stats', stats_node)

workflow.add_edge('stats', 'generator')
workflow.add_edge('generator', 'human_answer')
workflow.add_edge('human_answer', 'judge')
workflow.add_edge('hint_tool', 'human_answer') 
workflow.add_edge('human_next', 'stats')

workflow.add_conditional_edges(
    'judge',
    after_assesment_router,
    {
        'hint_tool': 'hint_tool',
        'parser': 'parser',
        '__end__': END
    }
)
workflow.add_conditional_edges(
    'parser',
    final_router,
    {
        'generator': 'human_next', 
        '__end__': END            
    }
)
memory = MemorySaver()

app = workflow.compile(
    checkpointer=memory,
    interrupt_before=['human_answer', 'human_next']
)

initial_state = {
    'hitPoints': 3,
    'score': 0,
    'tokens_so_far': 0,
    'topic': 'Computer Science',
    'difficulty': 'easy',
    'user_answers': [],
    'questions': [],
    'assessments': [],
    'hints': [],
    'need_to_score':3
}
 
def run_hitl():
    config = {'configurable': {"thread_id": "session_v1"}}
    app.invoke(initial_state, config=config)

    while True:
        snapshot = app.get_state(config)
        
        if not snapshot.next:
            print(10*'-')
            print("Game Over.")
            break

        next_node = snapshot.next[0] 

        if next_node == 'human_answer':
            print(10*'-')
            user_input = input("Type your answer: ")

            if user_input.lower() in ['q', 'quit']: break

            app.update_state(config, {'user_answers': [HumanMessage(content=user_input)]})
            print(10*'-')
            print("Thinking...")

            app.invoke(None, config=config)

        elif next_node == 'human_next':
            print(10*'-')
            user_input = input("Press [ENTER] for next question (or 'q' to quit)...")
            
            if user_input.lower() in ['q', 'quit']: break
            app.invoke(None, config=config)

        else:
            app.invoke(None, config=config)

        


# To Do:

# Figure out human in the loop - done
# what if I can use a chain? - done (with the custom wrapper)
# Implement tool calling!!! - done
# implement fix for hint calling - done
# implement prompt logic to avoid same questions - done

# finish implementation of 'manual next question'  - done
# implement a printing function across all nodes (rich lib ?) - will be done in separated please take a look at Antigravity branch

if __name__ == "__main__":
    run_hitl()
