from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI


class BaseAgent:

    @staticmethod
    def _make_prompt(user_template, system_prompt=False):
        prompt_list = []
        if system_prompt:
            prompt_list.append(
                (
                    "system",
                    system_prompt,
                )
            )
        prompt_list.append(
            (
                "human",
                user_template,
            )
        )
        return ChatPromptTemplate.from_messages(prompt_list)

    def structured_model(
        self,
        user_prompt,
        return_class,
        system_prompt=False,
        model="gpt-4o",
        include_raw=False,
        temperature=0,
        method="function_calling",
    ):
        prompt = self._make_prompt(user_prompt, system_prompt)
        llm = ChatOpenAI(model=model, temperature=temperature)
        llm = prompt | llm.with_structured_output(
            schema=return_class, include_raw=include_raw, method=method
        )
        return llm
