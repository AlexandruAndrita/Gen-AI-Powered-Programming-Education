system_message_nus = "You are an AI tutor. You have to help a student learning programming. The program uses Python. You have to strictly follow the format for the final output as instructed below."

user_message_nus_repair_basic = """
Following is the setup of a problem in Python. It contains the description and a sample testcase.

[Problem Starts]
{problem_data}
[Problem Ends]

Following is the student's buggy code for this problem:

[Buggy Code Starts]
{buggy_program}
[Buggy Code Ends]


Fix the buggy code. Output your entire fixed code between [FIXED] and [/FIXED].
"""

user_message_nus_hint_basic = """
Following is the setup of a problem in Python. It contains the description and a sample testcase.

[Problem Starts]
{problem_data}
[Problem Ends]

Following is the student's buggy code for this problem:

[Buggy Code Starts]
{buggy_program}
[Buggy Code Ends]

Provide a concise single-sentence hint to the student about one bug in the student's buggy code. Output your hint between [HINT] and [/HINT].
"""

user_message_nus_hint_using_repair = """
Following is the setup of a problem in Python. It contains the description and a sample testcase.

[Problem Starts]
{problem_data}
[Problem Ends]

Following is the student's buggy code for this problem:

[Buggy Code Starts]
{buggy_program}
[Buggy Code Ends]

Following is the repair of the students' buggy code for this problem:

[FIXED]
{repaired_program}
[/FIXED]

Based on the provided repair, provide a concise single-sentence hint to the student about one bug in the student's buggy code. Along the hint, please provide an explanation. The learner should read the explanation and understand what and why is wrong in his code. When providing the explanation, please do not describe the solution of the code. The role of the explanation is to understand the issue(s) in the buggy code. Output your hint and explanation between [HINT] and [/HINT]. Output your explanation between [EXPLANATION] and [/EXPLANATION].
"""

