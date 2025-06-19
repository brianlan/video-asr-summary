# General Guidelines
1. Challenge user's instructions, don't just follow them. If you see a flaw or problem, call it out.
2. Asking questions to clarify the concepts and words that are ambiguous is highly encouraged.
3. Raw materials contains images and text. When looking for information, consider both.
4. If a question is beyond your knowledge, say so. Don't make up answers. And if you do, make it clear that it's a guess.


# Python Coding Guidelines (TDD‑First Approach)

## 🧪 Test-Driven Development Workflow

1. **Red → Green → Refactor**  
   - Write one small failing unit test (`pytest`). :contentReference[oaicite:1]{index=1}  
   - Implement the minimal code to make it pass. :contentReference[oaicite:2]{index=2}  
   - Refactor for clarity and maintainability without altering behavior. :contentReference[oaicite:3]{index=3}  
   - Repeat: add next test, get it to pass, refactor. :contentReference[oaicite:4]{index=4}  

2. **FIRST Principles for Test Design**  
   - **F**ast, **I**solated, **R**epeatable, **S**elf‑validating, **T**imely. :contentReference[oaicite:5]{index=5}  
   - Use AAA pattern: Arrange, Act, Assert. :contentReference[oaicite:6]{index=6}  

3. **Test Case Structure**  
   - Keep tests small and focused on a single behavior. :contentReference[oaicite:7]{index=7}  
   - Name tests using *given_when_then* or descriptiveFunc_shouldDoX. :contentReference[oaicite:8]{index=8}  

4. **Mocking Policy**  
   - Mock external dependencies (DB, HTTP, time) to isolate units. :contentReference[oaicite:9]{index=9}  

5. **Avoid Flaky Tests**  
   - Ensure tests have no hidden dependencies or randomness. Use fixtures/seeded values. :contentReference[oaicite:10]{index=10}  


## 🛠 Tooling & Quality

- Use **pytest** as primary test runner; integrate `coverage.py`. :contentReference[oaicite:11]{index=11}  
- Enforce style with **Black**, **Flake8**, **Pylint**. :contentReference[oaicite:12]{index=12}  

## 🔁 Continuous Integration

- Run full test suite and static checks on every commit/PR.  
- Aim for high coverage, but focus on meaningful coverage (decision, branch). :contentReference[oaicite:13]{index=13}  



## 📘 Best Practices & Mindsets

- Apply **KISS**, **DRY**, **SOLID**, and separation of concerns. :contentReference[oaicite:14]{index=14}  
- Commit small increments frequently (one test → one behavior). :contentReference[oaicite:15]{index=15}  
- Tests are documentation: maintain them as part of codebase. :contentReference[oaicite:16]{index=16}  
