from typing import Any
import asyncio
from agents import Agent, GuardrailFunctionOutput, InputGuardrailTripwireTriggered, RunContextWrapper, Runner, TResponseInputItem, SQLiteSession, input_guardrail, trace
from dotenv import load_dotenv
from pydantic import BaseModel

load_dotenv(override=True)
    
async def main():
    
    class ProgrammingGuardrail(BaseModel):
        is_programming: bool
        
    guardrail_agent = Agent[Any](
        name="Guardrail Agent",
        instructions="Check if the user asks about anything besides programming and explaining code snippets.",
        output_type=ProgrammingGuardrail
    )
        
    @input_guardrail
    async def prog_guardrail(ctx: RunContextWrapper[None], agent: Agent, input: str | list[TResponseInputItem]) -> GuardrailFunctionOutput:
        result = await Runner.run(guardrail_agent, input, context=ctx.context)
        final_output = result.final_output_as(ProgrammingGuardrail)
        
        return GuardrailFunctionOutput(
            output_info=final_output,
            tripwire_triggered=not final_output.is_programming
        )
    
    code_explainer_agent = Agent[Any](
        name="Code Explainer Agent",
        instructions="You explain code snippets provided to you thoroughly and concisely. You point out errors if there are any, and give suggestions if needed.",
        model="gpt-4o-mini"
    )
    
    language_detecting_agent = Agent[Any](
        name="Language Detector",
        instructions="Your job is to tell the user in which programming language is the received code written in.",
        model="gpt-4o-mini"
    )
    
    normal_convo_agent = Agent[Any](
        name="Programming Agent",
        instructions="You answer questions about programming concisely and thoroughly.",
        model="gpt-4o-mini"
    )
    
    triage_agent = Agent[Any](
        name="Triage Agent",
        instructions="Your job is to provide the answer to the proposed question related to programming using tools that are provided to you. You will use the Code Explainer Agent to explain code snippets sent to you, and then you will use the Language Detector Agent to detect the programming language that the snippet is written in. You answer only after using both tools. If the user asks anything else thats not explaining code snippets, you will hand the convo off to Programming Agent! You only answer to questions related to programming!",
        model="gpt-4o-mini",
        tools=[
            code_explainer_agent.as_tool(tool_name="code_explainer", tool_description="explain code snippets"),
            language_detecting_agent.as_tool(tool_name="language_detector", tool_description="detect in what programming language the code snippet is in")
        ],
        handoffs=[normal_convo_agent],
        input_guardrails=[prog_guardrail]
    )
    
    with trace("Code Snippet Explainer"):
        
        session = SQLiteSession("programming_convo")
        
        print("Code Snippet Explainer\n")
        
        while True:
            try:
                code_snippet = input("")
                print("")
                
                result = await Runner.run(triage_agent, code_snippet, session=session)
                print(result.final_output)
                print("")
    
                user_input = input("Exit or Continue?: ")
                print("")
                if user_input.lower() in ["exit"]:
                    break
        
                continue_convo = await Runner.run(triage_agent, user_input, session=session)
                print(continue_convo.final_output)
                print("")
        
            except InputGuardrailTripwireTriggered:
                print("Guardrail tripwire triggered")
                tripwire = input("Exit: ")
                if tripwire.lower() in ["exit"]:
                    break
        
asyncio.run(main())