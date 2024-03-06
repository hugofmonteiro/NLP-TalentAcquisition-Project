# Talent Acquisition 

### Data Description:

The data comes from sourcing efforts. All fields that could directly reveal personal details were removed and given a unique identifier for each candidate. 

### Attributes:

- id : unique identifier for candidate (numeric)
- job_title : job title for candidate (text)
- location : geographical location for candidate (text)
- connections: number of connections candidate has, 500+ means over 500 (text)

#### Output (desired target):

- fit - how fit the candidate is for the role? (numeric, probability between 0-1)

#### Keywords: 

- “Aspiring human resources” or “seeking human resources”

### Goal:

- Predict how fit the candidate is based on their available information (variable fit)

### Success Metric:

- Rank candidates based on a fitness score
- Re-rank candidates when a candidate is starred

### Bonuses:

- We are interested in a robust algorithm, tell us how your solution works and show us how your ranking gets better with each starring action.
- How can we filter out candidates which in the first place should not be in this list?
- Can we determine a cut-off point that would work for other roles without losing high potential candidates?
- Do you have any ideas that we should explore so that we can even automate this procedure to prevent human bias?

#### Project code: 0kqGsJFyRmV9X2Le
