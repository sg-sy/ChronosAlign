system_prompt = """You are a one of the top researchers in the world creating a temporal dataset. The temporal dataset is aimed at capturing consistent and reoccuring events over time. I.e. event year, event date, subject, relationship, answer and question.

The relationship should be things such as where it was held, winner, second place, nationalities, films appeared in, movies produced, television shows appeared in, new players joining a team, venues and other easily verifiable outcomes and reltionships that can be answered in a couple of words. Where there is a table of results, filmography, discography, television shows or other table focus on that.

Provide enough detail in these information sets so that questions can be generated that lead to the answer specified. 

Make the answers as succinct as possible, but don't use abbreviations make sure to use the full name. If there are multiple answers to the same question provide them all as a list of string. 

Do not make things up, and only use the information provided in the content examined below.
Make sure to be comprehensive and provide detail on all years and dates. Limit to dates greater than the year 1999.

For the question field keep the question generic, don't specify a year, date or edition in that field. For example, in the case of a winner of the tour de france, set the question as "who was the winner of the tour de france?". For more specific questions, try and generalise. For example, instead of "what film did Russell Crowe win the Academy Award for Best Actor for?", set the question as "what film did the winner of the Academy Award for Best Actor star in?". Another approach to question making is using recent and latest, such as "what was Russel Crowe's latest film?". Alternatively, who recently joined or left a club, or released a new album, or a new film.
When considering answers that are set of split years use the first year as the event_year i.e 24/25 = 2024 or 2022/2023  = 2022.

output the question answer set in a json format with the following fields: event_year, event_date (if unavailable, set to None), subject, relationship, answer and question.
"""

user_prompt = """From the content below create a temporal dataset with as many questions sets as possible.
Content:
{content}"""


user_prompt_rephrase = """rephrase the following question to include the idea of recency, current or latest.
Some examples could be. 
- who was the winner of the tour de france? Rephrase: Who is the latest winner of the tour de france?
- who is the president of the united states of america? Rephrase: who is the current president of the united states of america?
- who came second in the giro di italia? Rephrase: who was the most recent person to place second at the giro do italia?
- where was the grammies held? Rephrase: where was the latest grammies held?

Switch between statements of latest, recent and current. write a single phrase. Do not provide a preamble, just write the rephrased statement after Rephrase:

Rephrase the following. {content}. Rephrase:"""


user_prompt_categories = """You are one of the worlds best natural language researchers. You are here to categorise certain questions into sport, culture or politics.
Output the answer as a single word.
Categorise the following: 
{content}
"""
system_prompt_update_questions = """You are a one of the top researchers in the world creating a temporal dataset. The temporal dataset is aimed at capturing consistent and reoccuring events over time. I.e. event year, event date, subject, relationship, answer and question.

The relationship should be things such as where it was held, winner, second place, nationalities, films appeared in, movies produced, television shows appeared in, new players joining a team, venues and other easily verifiable outcomes and reltionships that can be answered in a couple of words. Where there is a table of results, filmography, discography, television shows or other table focus on that.

Provide enough detail in these information sets so that questions can be generated that lead to the answer specified. 

Make the answers as succinct as possible, but don't use abbreviations make sure to use the full name. If there are multiple answers to the same question provide them all as a list of string. 

Do not make things up, and only use the information provided in the content examined below.
Make sure to be comprehensive and provide detail on all years and dates. Limit to dates greater than the year 1999.

For the question field keep the question generic, don't specify a year, date or edition in that field. For example, in the case of a winner of the tour de france, set the question as "who was the winner of the tour de france?". For more specific questions, try and generalise. For example, instead of "what film did Russell Crowe win the Academy Award for Best Actor for?", set the question as "what film did the winner of the Academy Award for Best Actor star in?". Another approach to question making is using recent and latest, such as "what was Russel Crowe's latest film?". Alternatively, who recently joined or left a club, or released a new album, or a new film.
When considering answers that are set of split years use the first year as the event_year i.e 24/25 = 2024 or 2022/2023  = 2022.
"""

user_prompt_update_questions = """From the content below update the answer for all question records that match the question.
Update the answer only, do not change the question or other fields. If there is no answer, set the answer to an empty string. Produce the output as a list of json objects with the exact schema of the question records, and ensure your produce the exact same number of records. 
Consider the question carefully. Where there are multiple answers to the same question, provide them all as a list of string.
Ensure that you consider the tabular patterns carefully to assign the correct answer(s) to the year specified.
**Always produce a record for each question and event_year combination**
**provide the JSON only, do not provide any other text or explanation**
**For the json keys, use double quotes**
**Make sure to consider incumbent positions. If the period specifies a start year and has an end that is either incumbent. present or open ended, consider and reason about the answer ensuring that it can answer for the years present below* The current year is 2026*
Questions:
{question}
Content:
{content}"""
