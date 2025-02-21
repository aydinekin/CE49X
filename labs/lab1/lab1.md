
# Lab 1: Traffic Data Analysis (due Thursday, Feb 20th, 5pm)

**Objective:**  
In this lab, you will practice the basics of Python programming by loading a small traffic dataset, computing basic descriptive statistics, and then committing and pushing your work to GitHub.

---

## 1. Overview

In this lab, you will:

- Load a CSV dataset containing traffic information.
- Use Python (and the Pandas library) to compute:
  - Minimum vehicle count
  - Maximum vehicle count
  - Mean (average) vehicle count
- Print these statistics to the console.
- Practice using Git to commit your code and push your changes to your GitHub repository.

---

## 2. Prerequisites

Make sure you have completed the following setup before starting Lab 1:

- **Python & Anaconda/Miniconda Installed:**  
  Ensure you have Python 3.10 or later available in your environment.

- **Pandas Installed:**  
  If you have not installed Pandas yet, you can do so by running:
  ```bash
  pip install pandas
  ```

- **Git & GitHub Setup:**  
  Confirm that you have Git installed and your GitHub account is set up. You should have already cloned or forked the course repository.

---

## 3. The Dataset

A sample dataset file named `traffic_data.csv` is provided in the `/datasets/` folder. A sample of its contents is shown below:

```csv
timestamp,vehicle_count,location
2025-01-01 08:00,15,Main Street
2025-01-01 09:00,20,Main Street
2025-01-01 10:00,12,Main Street
2025-01-01 11:00,18,Main Street
```

> **Note:** If you do not see the file in your local repository, please pull the latest changes.

---

## 4. The Python Script

Create a new Python script file (e.g., `lab1_traffic_analysis.py`) in the `/labs/lab1/` folder. Use the code snippet below as a starting point.

```python
# lab1_traffic_analysis.py

import pandas as pd

def main():
    # Load the traffic dataset
    try:
        df = pd.read_csv('../../datasets/traffic_data.csv')
    except FileNotFoundError:
        print("Error: The dataset file was not found. Please ensure 'traffic_data.csv' is located in the /datasets/ folder.")
        return

    # Compute basic descriptive statistics for 'vehicle_count'
    min_value = df['vehicle_count'].min()
    max_value = df['vehicle_count'].max()
    mean_value = df['vehicle_count'].mean()

    # Print the results
    print("Traffic Data Analysis:")
    print(f"Minimum vehicle count: {min_value}")
    print(f"Maximum vehicle count: {max_value}")
    print(f"Mean vehicle count: {mean_value:.2f}")

if __name__ == '__main__':
    main()
```

> **Tips:**
>
> - Make sure the path to your CSV file is correct. Adjust the path (`../../datasets/traffic_data.csv`) if necessary, depending on your repository structure.
> - You can add additional print statements or error handling as needed.

---

## 5. Running Your Script

To run your script, open a terminal in the `labs/lab1/` directory and type:

```bash
python lab1_traffic_analysis.py
```

You should see output similar to:

```
Traffic Data Analysis:
Minimum vehicle count: 12
Maximum vehicle count: 20
Mean vehicle count: 16.25
```

*(The exact values will depend on the contents of your `traffic_data.csv` file.)*

---

## 6. Version Control: Committing and Pushing Your Code

Follow these steps to commit your lab work and push it to GitHub:

1. **Check your repository status:**

   ```bash
   git status
   ```

2. **Stage your changes:**

   ```bash
   git add labs/lab1/lab1_traffic_analysis.py
   ```

3. **Commit your changes with a descriptive message:**

   ```bash
   git commit -m "Lab 1: Add traffic data analysis script"
   ```

4. **Push your changes to GitHub:**

   ```bash
   git push origin main
   ```

> **Note:** If you are working on a separate branch (recommended for labs), make sure you push that branch instead. For example:
>
> ```bash
> git push origin lab1-traffic-analysis
> ```

---

## 7. Submission

- Ensure that your script runs without errors.
- Verify that your code is pushed to your GitHub repository.
- Email the URL of your GitHub repository to eyuphan.koc@bogazici.edu.tr.
---

## 8. Additional Resources

- **Pandas Documentation:** [https://pandas.pydata.org/docs/](https://pandas.pydata.org/docs/)
- **GitHub Guides:** [https://guides.github.com/](https://guides.github.com/)

---

Good luck with Lab 1!
