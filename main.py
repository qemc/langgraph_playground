from langgraph.graph.state import StateGraph, END
from langchain_core.runnables.base import RunnableLambda
import dotenv
from langchain_core.tools import tool
from typing import Annotated, TypedDict, List
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import PydanticOutputParser
from langgraph.checkpoint.memory import MemorySaver

dotenv.load_dotenv()

llm = ChatOpenAI(model = 'gpt-5-nano')

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

        if not raw_text:
            raise ValueError(f"Router error with prompt: {prompts}")


        tokens_used = 0
        if hasattr(response, "usage_metadata") and response.usage_metadata:
            tokens_used = response.usage_metadata.get("total_tokens", 0)

        elif hasattr(response, "response_metadata"):
             token_usage = response.response_metadata.get("token_usage", {})
             tokens_used = token_usage.get("total_tokens", 0)

        prev_tokens = input.get("tokens_so_far", 0)
        new_total_tokens = tokens_used + prev_tokens

        #print(raw_text)

        return {
            **input,
            output_key: raw_text,
            "tokens_so_far": new_total_tokens,
            "prompt_messages": prompts
        }

    return RunnableLambda(_func)

class JudgeAnswer(BaseModel):
    result: str = Field(description='the result of the answer assesment, can be only True, False or Tool') # TypeScript :(((
    just: str = Field(description='the justification of the result. The the judge resturned that result?')

@tool
def get_hint_tool(question: str, topic: str):

    '''
    A tool to receive a hint when the user asks for it.
    '''

    prompt = system_user_prompt(
        system_prompt='''
        You are a helpful assistant that provides hits to the question mentioned in context.
        You never answer directly and indirectly to the question provided. 
        The hint should be obvious to the user.
        The hint should be 1-2 sentences and only this. 
        You should return a normal string with the hint without any additional information. 
        The question is related to the {topic}
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

    return str(res.get('resp')).strip()
    
tools = [get_hint_tool]

class QuizzState(TypedDict):
    hitPoints: int
    score: int
    tokens_so_far: float
    topic: str
    difficulty: str
    user_answers: Annotated[List[BaseMessage], add_messages]
    questions: Annotated[List[BaseMessage], add_messages]
    assesments: List[JudgeAnswer]


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
            4. Wait for the user to reply.:
        ''', 
        user_prompt='''

            Generate questions according to system prompt.
        ''' 
    )
    
    topic = state['topic']
    difficulty = state['difficulty']
    tokens_so_far = state['tokens_so_far']
    
    prompt = create_prompt(llm,'resp', prompt)
    
    result = prompt.invoke({
        "topic": topic,
        "difficulty": difficulty
    })
    question = str(result.get("resp")).strip()
    tokens_used = str(result.get("tokens_so_far")).strip()

    new_token_so_far = int(tokens_so_far) + int(tokens_used)

    
    print(question)
    
    return {
        'questions': [question],
        'tokens_so_far': new_token_so_far
    }

def human_node(state:TypedDict):
    pass


def judge_node(state: TypedDict):

    llm_with_tools = llm.bind_tools(tools)

    prompt = system_user_prompt(
        system_prompt='''
        You are a judge whos job is to assess the truth of the answer provided by 
        the user to the question.

        You are an expert on the following topic: {topic} 

        Both the question and the answer are provided in the contect within
        the user prompt.

        Ypour answer should be provided in json format (it needs
        to be correct, later on it will be parsed to the dict object.)
        The format:
        {{
            "result": "true" # or false, depends on the user answer. Only one word
            "just": "here you provide your justification of the assesment." # If this was true, you explain why, and if false you explain why.        
        }}
        Example input:
        Question:
        In a standard football match, how many players are on the field for one 
        team at the start of the game (excluding substitutions)?
        A) 9
        B) 10
        C) 11
        D) 12

        User answer:
        9

        Your answer would be:
        {{
            "result": "false" 
            "just": "The football team contains 11 players. 10 in field, and one goalkeeper"
        }}
        ''',
        user_prompt='''
        Question:
        {question}
        User asnwer:
        {user_answer}
        '''
    )

    topic = state['topic']
    tokens_so_far = state['tokens_so_far']

    prompt = create_prompt(llm_with_tools,'resp', prompt)
    

    question = state['questions'][-1]
    user_answer = state['user_answers'][-1]

    result = prompt.invoke({
        "topic": topic,
        "user_answer": user_answer,
        "question":question
    })

    tokens_used = str(result.get("tokens_so_far")).strip()

    new_token_so_far = int(tokens_so_far) + int(tokens_used)

    assesment = result.get("resp")
    current_assesments = state.get('assesments', [])

    return {
        'assesments': current_assesments + [assesment],
        'tokens_so_far': new_token_so_far
    }


workflow = StateGraph(QuizzState)

workflow.add_node('generator',quiz_generator_node)
workflow.add_node('judge',judge_node)
workflow.add_node('human', human_node)

workflow.set_entry_point('generator')

workflow.add_edge('generator', 'human')
workflow.add_edge('human', 'judge')

workflow.add_edge('judge', END)

memory = MemorySaver()


app = workflow.compile(
    checkpointer=memory,
    interrupt_before=['human']
)

initial_state = {
    'hitPoints': 3,
    'score': 0,
    'tokens_so_far': 0,
    'topic': 'football',
    'difficulty': 'easy',
    'user_answers': [],
    'questions': [],
    'assesments': []
}


def run_hitl():

    config = {'configurable': {"thread_id": "session1"}}
    app.invoke(initial_state, config = config)

    while True:

        snapshot = app.get_state(config)

        if not snapshot.next:
            print("Game over")
            break
        
        user_input = input("Type your answer: ")

        if user_input.lower() in ['q', 'quit']:
            print("Exiting...")
            break

        human_msg = HumanMessage(content=user_input)
        app.update_state(config, {'user_answers': [human_msg]})

        app.invoke(None, config=config)


# To Do:
# Figure out human in the loop - done
# what if I can use a chain?   - done (with the custom wrapper)
# Implement tool calling!!!


if __name__ == "__main__":
    run_hitl()
