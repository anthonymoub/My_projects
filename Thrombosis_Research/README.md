[![Open in Visual Studio Code](https://classroom.github.com/assets/open-in-vscode-c66648af7eb3fe8bc4f294546bfd86ef473780cde1dea487d3c4ff354943c9ae.svg)](https://classroom.github.com/online_ide?assignment_repo_id=10030286&assignment_repo_type=AssignmentRepo)
# Homework-2: 

Visual Exploratory Data Analysis and professional report generation

**IMPORTANT:**

* This is a long assignment. Give yourself ample time to complete it. 
* Data wrangling and EDA can be very  time consuming. 
* You will not be able to complete this if you start a day or two before it is due. 

* There are many requirements and details. Please read through the assignment, in it's entirety, before beginning. 

## Overview 

Imagine that you were working as a data scientist at a consulting firm. An external entity (client) has paid a significant amount of money for your firm to make sense of a medical data set. They plan to incorporate your findings to help guide the allocation of a multi-million dollar research grant. They want the results in two weeks, after which point the contract terminates and your consulting firm will move onto a new unrelated project.

You will perform visual and non-visual exploratory analysis to better understand the shape & structure of the data, investigate initial questions, and develop preliminary insights & hypotheses for the client. Your final submission will take the form of multiple reports consisting of captioned visualizations that convey key insights gained during your analysis.

Your boss wants two documents, within a two week time-horizon. 

1) **Section-01:** (`EDA.html` ) This will be a technical exploration notebook, documenting your EDA process, from start to finish. It will be shared with other data scientist at your firm, to check your work and verify your methodology and conclusions. It should be reproducible, meaning that all code is present, and the product should be well documented and intuitive to an outsider. The plots in the section do not necessarily have to be publication quality, they are meant for your team not the client, however, the entire document should flow well and be logically constructed, with sufficient descriptive markdown documentation and proper commenting for an outsider.
   1) **NOTE**: Your exploration notebook will likely change significantly throughout the EDA process. What do you include at the end should be the relevant code, needed to reproduce your process and arrive at your final conclusions. You do not necessarily need to include every single thing that you tried, only what strengthens your final conclusions.

2) **Section-02:** (`Results.html`) A Concise, 2 or 3 page summary report of your most important results, to be shared with the non-technical clients. This should summarize your finding and should include at least **FOUR** publication quality plots.

**Is an very open ended assignment, here is a one sentence summary of it:** Thoroughly explore the provided data-set visually to extract as much meaningful information as possible in the provided time, and formulate your findings in a professional report

## Required software usage: 

1) You can use either R or Python for this assignment, it is your choice. However, it is recommended that you do it in both for practice.
2) **Report generation** 
   1) YOU MUST present your findings in the form of a published Quarto document. 
      1) You can generate the report in either `.ipynb`, or `.qmd`. However, it needs to be rendered using Quarto with the appropriate YAML header lines to generate a clean and polished professional document (see lab-2.1 for a set of relevant examples, any of these are suitable templates)
      2) **Note:**
         1) With Quarto’s `convert` command, you can re-format and jump between the different file-formats. 
         2) `quarto convert HW-2.rmd` will convert the file to `HW-2.ipynb`
         3) `quarto convert HW-2.qmd` will convert the file to `HW-2.ipynb`
         4) `quarto convert HW-2.ipynb` will convert the file to `HW-2.qmd`, which can be renamed `HW-2.rmd` or just opened in R-studio like any other `rmd` file, just like normal.
         5) `quarto render HW-2.ipynb` , `quarto render HW-2.rmd`, or `quarto render HW-2.qmd` will render the notebook  into an aesthetically pleasing output, provided the correct YAML lines.
3) **Visualization tools** 
   1) Your visualizations should be **static visualizations**, and not dynamic/interactive visualizations (no Javascript), nor require any running R or Python server to see (no Shiny/Dash/Streamlit)
   2) In this assignment, the **only viz tools** you can use are **ggplot2** and its extensions (in R) and **matplotlib/seaborn** (in Python). 
   3) **Prohibited software:** No tableau or visualization that uses *dynamic visualization libraries* like **plotly, D3.js, Vega, htmlwidgets** or similar packages. Dynamic visualization libraries, or server-side tools like Shiny, Dash, Streamlit, etc, are **not allowed for this assignment**. If in doubt, ask. 
   4) The Reason for this is that we want you to master `ggplot` and `MPL`, also we will cover interactivity later in the course.

## Submission 

- You need to upload TWO things to Canvas when you are done.
  - The Canvas upload will act as the timestamp for your submission
  - A FULLY self contained HTML file for `EDA.html`  
    - use `embed-resources: true`  in the YAML header to make it self-contained 
  - A FULLY self contained HTML file for `Results.html`  
  - The URL to the GitHub repository hosting your code MUST be included in both  `EDA.html`  and  `Results.html` 
  - **IMPORTANT**: All code, outputs, and visualizations should be accessible. However, "on the job" the code would not be included in a product intended for a non-technical audience. 
- The final uploaded version should NOT have any code-errors present.
- All outputs must be visible in the uploaded version, including code-cell outputs, images, graphs, etc

## Assignment details

### Step-0: Familiarize yourself with the data 

The data files are stored in the `data/` folder. A description of this data and are given in `medical-data-description.pdf`.  

In a section called `Data summary` in `EDA.html`, provide the following

Include a short, two or three paragraph synopsis, IN YOUR OWN WORDS, based on your reading of this document.

**WARNING:** DO NOT go overboard on review of subject matter expertise. It's very common in data science that you need to quickly get up to speed with the fundamental concepts behind a data set. However, it is important to not spend an excessive amount of time on this. Your goal in this assignment is EDA, not to become a biologist.

### Step-1: Initial questions

In a section called `Initial questions` in `EDA.html`, provide the following

* *Prior to analysis* — you should write down an initial set of **AT LEAST three different questions** you’d like to investigate. 
* Note that the stated goals in the data description **cannot be used as your questions**, since the goals are abstract and general. The questions you want to investigate need to be more specific and created by you.

### Step-2: Data Ingestion, cleaning, and munging

You will need to ingest this data into R or Python and prepare it for exploratory visual analysis. 

You will create an _analytic data set_ formed by appropriately joining the data in the 3 provided files. This data **must be tidy for the patient unit of analysis**, so there should be no data in the column headers, nor should there be any duplication of rows , among other properties that ensure the data is tidy. The column names must be in **[snake_case](https://en.wikipedia.org/wiki/Snake_case)** format, all lower case, or a format aligned with the code style you are following. 

Do this in a section called `Data munging` in `EDA.html`. You are welcome to save intermediate results into additional files as needed and reload them later as part of the visualization phase.

### Step-3: Exploratory Visual Analysis

Next, you will perform an exploratory analysis of this dataset using R or Python. Do this in a section called `Exploratory analysis` in `EDA.html`. 

You should consider rapid prototyping rather than final presentation during your exploration phase. Consider *two different phases* of exploration.

In the first phase, you should seek to *gain an overview* of the shape & structure of your dataset. What variables does the dataset contain? How are they distributed? Are there any notable data quality issues? Are there any surprising relationships among the variables? Be sure to also perform “sanity checks” for patterns you expect to see! This section includes both visual and non-visual EDA

In the second phase, you should investigate your initial questions, as well as *any new questions* that arise during your exploration. For each question, start by creating a visualization that might provide a useful answer. Then refine the visualization (e.g., by adding additional variables, changing sorting or axis scales, transforming your data by filtering or subsetting it, etc.) to develop better perspectives, explore unexpected observations, or sanity check your assumptions. You should repeat this process for each of your questions, but feel free to revise your questions or branch off to explore new questions if the data warrants.

The goal of this assignment is not modeling, however you are welcome to include modeling methods in your technical write up. However they should not be the main focus of the work. 

### Step-4: Generate publication quality plots

You'll need to create **FOUR** publication quality plots for inclusion in `Results.html` (see below).

These should be generated in a section called `Final plots` in `EDA.html` and exported to files for inclusion  in `Results.html` . Use the naming convention `plot-01.png` ,  `plot-02.png`  ... etc. 

The code used to generate these plots and the plot outputs should also be visible in the `EDA.html` file.

These plots should support and strengthen your most important conclusions from the EDA process.

Each visualization image should be generated using R or Python code, have appropriate titles, labels and annotations,  and exported to an appropriate image format (png/jpg/pdf/svg). 

It is recommended, but not required, to annotate your images to draw attention to specific features of the data. 

### Step-5: Technical summary 

Finally you need to document your process in a section called `Technical summary` in `EDA.html`. 

This section should be around 350 to 500 words long. 

The end of your technical summary should include a brief summary of main lessons learned, as well as commentary on what kinds of visualizations were useful and what kinds weren't for answering your questions. These could refer to figures in the report and/or other figures that you may have tried but didn't include in the report. 

 Your “insights” can also  include important surprises or issues (such as data quality problems affecting your analysis, missing data patterns) as well as responses to your analysis questions. 

Would you say that the data said created by the authors is good? What kinds of problems does it have, if any?

### Step-6: Non-technical report

The final submission should take the form of a non-technical report  in `Results.html`  which consists of **4 or more** captioned **static visualizations** detailing your most important insights, *which you want to share with the client*.

This section should be around 400 to 800 words long, with around 2 to 3 pages including figures. 

**This part of the submission is more about storytelling. It should help non-technical shareholders makes sense of a highly technical data set.**

It should be a **well-formatted and styled report** using headers, paragraphs, the generated figures and captions, and any other features that increase the readability and navigability of the report for the reader. In other words, for the reader, the file should be easy to navigate, to move between sections or figures, and should be easy to read, both in terms of writing and in terms of visual presentation and experience, by incorporating format and style into the text and the overall page.

In your report, **each figure must be accompanied with a title and descriptive caption (2-4 sentences long) describing the insight(s) learned from that view**.  Provide sufficient detail for each caption such that anyone could read through your figure and caption  and understand what you’ve learned from the figure. 

**The figures will be linked and contextualized through paragraph(s) of narrative content describing your questions and how you are finding answers to them**. For example, "In fig-2, we can see that .... which implies that .... " 

## Additional requirements 

1. **Proper use of git and Github**

   +  No direct uploads to Github via a web-browser are allowed
   +  Everything staged, committed, and pushed from your local repo 
   +  REGULAR commits and pushes to the cloud
   +  [descriptive](https://www.freecodecamp.org/news/how-to-write-better-git-commit-messages/) [commit](https://www.freecodecamp.org/news/writing-good-commit-messages-a-practical-guide/) [messages](https://reflectoring.io/meaningful-commit-messages/) (or at least, attempts to do so)
   +  `.gitignore` has all the necessary files to make the submission clean.
   +  Any intermediate data sets you may create must not be committed. Include them in the `.gitignore` file.

2. **Code formatting**

   * Properly formatted code is the hallmark of a competent and professional programmer and team player, and is often _required_ in corporate environments. It makes reading code (your own and others) much much easier and more comprehensible, since the code becomes stylistically similar and you spend less time deciphering the code and more time actually coding and understanding code. This, in turn, lowers the effort required for code reviews and updates (and grading!).  The [tidyverse style](https://style.tidyverse.org) for R and the [pep8 style](https://pep8.org) for Python are widely used standards. _You should read them at least once to have an idea of what's expected in the style_. However, you **don't have to maintain these styles by hand!**.
   * Your code **must be properly formatted** using a standard style: the [tidyverse style](https://style.tidyverse.org) for `R`, and [pep8](https://pep8.org) for `Python`. It should also have *sections and descriptive comments* to aid the reader's understanding of the code and align the code to the various deliverables. Variable names and column names in data frames should be descriptive of the data they contain (no `x`, `V2`, etc. ) and also should be in a format consistent with the code style you are using. (see below). 
   * For R, the **[styler](https://styler.r-lib.org)** package auto-formats R and RMarkdown files to the tidyverse standard. Installing it also installs an add-in to RStudio that makes it even easier to run. The **[lintr](https://github.com/r-lib/lintr)** package does _static code analysis_ and points out style, syntax and some semantic issues with your code, but doesn't modify your file. It also can be baked in to RStudio.
   * For Python, there are a couple of popular auto-formatters. The first, **[black](https://pypi.org/project/black/)**, takes a dictatorial view of formatting and doesn't allow much customization, but does its job beautifully, conforming to the pep8 standard and beyond. It also **[integrates with most editors](https://black.readthedocs.io/en/stable/integrations/editors.html)**.   The other, **[autopep8](https://pypi.org/project/autopep8/)** auto-formats Python script files to the pep8 standard. It also can be integrated into editors, but you need Google for the instructions. You can also do static code analysis using **[pylint](https://pylint.org)** and **[flake8](https://flake8.pycqa.org/en/latest/)**, which covers a broader range of problems with your code.

   * In VS-code, you can use `black` to format an `.ipynb`  with the following commands

     - MacOS: `Option+Shift+F`

     - Windows: `Alt+Shift+F`

     * To automatically reformat the code cells in the notebook to conform with the standards mentioned above.

     * This will instruct you to download the relevant packages the first time you run these command.
