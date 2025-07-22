# General Guidelines
1. Challenge user's instructions, don't just follow them. If you see a flaw or problem, call it out.
2. Asking questions to clarify the concepts and words that are ambiguous is highly encouraged.
3. Raw materials contains images and text. When looking for information, consider both.
4. If a question is beyond your knowledge, say so. Don't make up answers. And if you do, make it clear that it's a guess.
5. Before you start writing code, do create a plan or sub-task list. And when a sub-task is done, do not forget to check it off.

# Coding Guidelines

1. Strictly follow the TDD-First Approach
  - Write a failing test that simulates user or API interaction (`pytest`, etc.).
  - Write the minimal code to make the test pass.
  - Refactor to improve readability, maintainability, or modularity.
  - Repeat: one test → working feature → refactor cycle.
2. Continuous Integration
  - Automate tests and checks on every commit and pull request.
  - Use version control well: Descriptive commit messages. Small commits (1 change → 1 commit).
3. Code Quality
  - Follow **KISS**, **DRY**, **SOLID**, **YAGNI**.
  - Apply meaningful code comments; avoid explaining *what*, explain *why*.
