import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import os
import io
import openai
from base_agent import BaseAgent, AgentConfig, PipelineState

# Helper function for Cramer's V
def cramers_v(x, y):
    """Calculates Cramer's V statistic for categorical-categorical association."""
    confusion_matrix = pd.crosstab(x, y)
    chi2 = stats.chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2 / n
    r, k = confusion_matrix.shape
    
    if n == 0:
        return 0
    
    phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))
    rcorr = r - ((r-1)**2)/(n-1) if n > 1 else r
    kcorr = k - ((k-1)**2)/(n-1) if n > 1 else k
    
    if min((kcorr-1), (rcorr-1)) == 0:
        return 0
    
    return np.sqrt(phi2corr / min((kcorr-1), (rcorr-1)))

class EDAAnalyzer:
    """
    A class to perform a deep Exploratory Data Analysis (EDA) on a pandas DataFrame.
    """
    
    def __init__(self, df, schema, target_col=None, ordinal_cols=None, generate_plots=True, output_format='md', plot_style='seaborn', openai_api_key=None):
        # ✅ FIXED: Direct parameters, no state access
        self.df = df
        self.schema = schema
        self.target_col = target_col
        self.ordinal_cols = ordinal_cols if ordinal_cols is not None else []
        self.generate_plots = generate_plots
        self.output_format = output_format
        self.plot_style = plot_style
        self.openai_api_key = openai_api_key
        
        if self.df is None:
            raise ValueError("No data available for EDA")
        if self.schema is None:
            raise ValueError("No schema available for EDA")
            
        self.numerical_cols = [col for col, type in self.schema.items() if type == 'numerical']
        self.categorical_cols = [col for col, type in self.schema.items() if type == 'categorical' and col != self.target_col]
        self.nominal_cols = [col for col in self.categorical_cols if col not in self.ordinal_cols]
        
        self.output_dir = "eda_report"
        self.plots_dir = os.path.join(self.output_dir, "plots")
        self.report_path = os.path.join(self.output_dir, "eda_report.md")
        
        self.results = {
            "overview": [],
            "continuous": [],
            "categorical": [],
            "bivariate": [],
            "ai_summary": ""
        }
        
        if not os.path.exists(self.plots_dir):
            os.makedirs(self.plots_dir)
            
        # ✅ FIXED: Removed hardcoded API key - security issue!
        if self.openai_api_key and len(self.openai_api_key) > 20:  # Basic validation
            try:
                self.client = openai.OpenAI(api_key=self.openai_api_key)
                print("OpenAI API configured. AI summary will be generated.")
            except Exception as e:
                print(f"OpenAI API configuration failed: {e}")
                self.client = None
        else:
            print("OPENAI_API_KEY not configured or invalid. AI summary will be skipped.")
            self.client = None

    def run_analysis(self):
        """Main analysis method that coordinates all EDA steps."""
        self._analyze_overview()
        self._analyze_continuous_variables()
        self._analyze_categorical_variables()
        
        if self.target_col and self.target_col in self.df.columns:
            self._analyze_bivariate_relationships()
        elif self.target_col:
            print(f"Warning: Target column '{self.target_col}' not found in data. Skipping bivariate analysis.")
            
        self.results["ai_summary"] = self._generate_llm_summary()
        self.generate_report()

    def _save_plot(self, fig, filename):
        path = os.path.join(self.plots_dir, filename)
        fig.savefig(path, bbox_inches='tight')
        plt.close(fig)
        return os.path.join("plots", filename)

    def _generate_llm_summary(self):
        """Generates a summary of the EDA findings using an LLM."""
        if self.client is None:
            return "AI summary could not be generated because the OpenAI API key is not configured."
            
        print("Generating AI summary...")
        prompt = "You are an expert data analyst. Based on the following exploratory data analysis results, provide a high-level summary of the key insights in Markdown format. Focus on the most significant findings regarding data quality, variable distributions, and relationships between variables.\n\n"
        
        # Add overview results
        prompt += "## Data Overview\n"
        for title, content in self.results["overview"]:
            prompt += f"### {title}\n{content}\n\n"
            
        # Add continuous variable insights
        prompt += "## Continuous Variables\n"
        for result in self.results["continuous"]:
            if "col" in result:
                prompt += f"- {result['col']}: Distribution analyzed\n"
                
        # Add categorical variable insights
        prompt += "## Categorical Variables\n"
        for result in self.results["categorical"]:
            if "col" in result:
                prompt += f"- {result['col']}: Frequency distribution analyzed\n"
                
        # Add bivariate insights if available
        if self.target_col and self.results["bivariate"]:
            prompt += f"## Relationships with Target ({self.target_col})\n"
            for result in self.results["bivariate"]:
                if "col" in result:
                    prompt += f"- {result['col']}: Association analyzed\n"
        
        prompt += "\nPlease provide a concise, professional summary highlighting the most important findings."
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are an expert data analyst."},
                    {"role": "user", "content": prompt}
                ]
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error calling OpenAI API: {e}")
            return f"AI summary generation failed due to an API error: {e}"

    def _analyze_overview(self):
        """Analyzes basic data overview including info, stats, and missing values."""
        print("Analyzing data overview...")
        
        # Basic information
        buffer = io.StringIO()
        self.df.info(buf=buffer)
        self.results["overview"].append(("Data Info", f"<pre>\n{buffer.getvalue()}\n</pre>"))
        
        # Descriptive statistics
        self.results["overview"].append(("Descriptive Statistics", self.df.describe().to_markdown()))
        
        # Missing values analysis
        missing = self.df.isnull().sum()
        missing = missing[missing > 0]
        if not missing.empty:
            self.results["overview"].append(("Missing Values", missing.to_frame('count').to_markdown()))
        else:
            self.results["overview"].append(("Missing Values", "No missing values found."))

    def _analyze_continuous_variables(self):
        """Analyzes continuous/numerical variables."""
        print("Analyzing continuous variables...")
        
        for col in self.numerical_cols:
            if col not in self.df.columns:
                print(f"Warning: Numerical column '{col}' not found in data. Skipping.")
                continue
                
            fig, ax = plt.subplots(1, 2, figsize=(12, 5))
            
            # Histogram with KDE
            sns.histplot(self.df[col], kde=True, ax=ax[0])
            ax[0].set_title(f'Histogram of {col}')
            
            # Boxplot
            sns.boxplot(x=self.df[col], ax=ax[1])
            ax[1].set_title(f'Boxplot of {col}')
            
            plot_path = self._save_plot(fig, f'dist_{col}.png')
            self.results["continuous"].append({
                "col": col,
                "plot_path": plot_path
            })
        
        # Correlation heatmap for multiple numerical variables
        if len(self.numerical_cols) > 1:
            available_numerical = [col for col in self.numerical_cols if col in self.df.columns]
            if len(available_numerical) > 1:
                corr = self.df[available_numerical].corr(method='pearson')
                fig, ax = plt.subplots(figsize=(10, 8))
                sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', ax=ax)
                ax.set_title('Pearson Correlation Heatmap')
                
                heatmap_path = self._save_plot(fig, 'pearson_heatmap.png')
                self.results["continuous"].append({
                    "type": "heatmap",
                    "path": heatmap_path
                })

    def _analyze_categorical_variables(self):
        """Analyzes categorical variables."""
        print("Analyzing categorical variables...")
        
        all_cats = self.nominal_cols + self.ordinal_cols
        
        for col in all_cats:
            if col not in self.df.columns:
                print(f"Warning: Categorical column '{col}' not found in data. Skipping.")
                continue
                
            # Frequency plot
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.countplot(y=self.df[col], order=self.df[col].value_counts().index, ax=ax)
            ax.set_title(f'Frequency of {col}')
            
            bar_path = self._save_plot(fig, f'bar_{col}.png')
            
            # Goodness of fit test (chi-square)
            gof_stat, gof_p = stats.chisquare(self.df[col].value_counts())
            
            self.results["categorical"].append({
                "col": col,
                "plot_path": bar_path,
                "gof_test": {"stat": gof_stat, "p": gof_p}
            })

    def _analyze_bivariate_relationships(self):
        """Analyzes relationships with target variable."""
        print("Analyzing bivariate relationships...")
        
        target = self.target_col
        
        if target not in self.df.columns:
            print(f"Warning: Target column '{target}' not found in data. Skipping bivariate analysis.")
            return
        
        # Continuous vs Target
        for col in self.numerical_cols:
            if col not in self.df.columns:
                continue
                
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.boxplot(x=self.df[target], y=self.df[col], ax=ax)
            ax.set_title(f'{col} vs. {target}')
            
            plot_path = self._save_plot(fig, f'bivariate_box_{col}.png')
            
            # Statistical test
            groups = [g[col].dropna() for _, g in self.df.groupby(target)]
            if len(groups) == 2:
                test_name = "T-test"
                stat, p = stats.ttest_ind(*groups)
            else:
                test_name = "ANOVA"
                stat, p = stats.f_oneway(*groups)
            
            self.results["bivariate"].append({
                "type": "continuous",
                "col": col,
                "plot_path": plot_path,
                "test": {"name": test_name, "stat": stat, "p": p}
            })
        
        # Categorical vs Target
        all_cats = self.nominal_cols + self.ordinal_cols
        for col in all_cats:
            if col not in self.df.columns:
                continue
                
            crosstab = pd.crosstab(self.df[col], self.df[target])
            chi2, p, _, _ = stats.chi2_contingency(crosstab)
            v = cramers_v(self.df[col], self.df[target])
            
            self.results["bivariate"].append({
                "type": "categorical",
                "col": col,
                "crosstab": crosstab.to_markdown(),
                "test": {
                    "name": "Chi-Square Test",
                    "stat": chi2,
                    "p": p,
                    "cramers_v": v
                }
            })

    def generate_report(self):
        """Generates a Markdown report of the findings."""
        print("Generating report...")
        
        with open(self.report_path, 'w') as f:
            f.write("# Exploratory Data Analysis Report\n\n")
            
            # AI Summary
            if self.results["ai_summary"]:
                f.write("## 1. AI-Generated Summary\n\n")
                f.write(self.results["ai_summary"] + "\n\n")
            
            # Data Overview
            f.write("## 2. Data Overview\n\n")
            for title, content in self.results["overview"]:
                f.write(f"### {title}\n")
                f.write(f"{content}\n\n")
            
            # Continuous Variables
            f.write("## 3. Continuous Variable Analysis\n\n")
            for result in self.results["continuous"]:
                if result.get("type") == "heatmap":
                    f.write("### Correlation Heatmap\n")
                    f.write(f"![Correlation Heatmap]({result['path']})\n\n")
                else:
                    col = result['col']
                    f.write(f"### Analysis of '{col}'\n")
                    f.write(f"![Distribution of {col}]({result['plot_path']})\n\n")
            
            # Categorical Variables
            f.write("## 4. Categorical Variable Analysis\n\n")
            for result in self.results["categorical"]:
                col = result['col']
                f.write(f"### Analysis of '{col}'\n")
                f.write(f"![Frequency of {col}]({result['plot_path']})\n\n")
                
                gof_test = result['gof_test']
                f.write("**Goodness of Fit Test (vs. Uniform Distribution):**\n")
                f.write(f"- Chi-Square Statistic: {gof_test['stat']:.2f}\n")
                f.write(f"- P-value: {gof_test['p']:.3f}\n")
                f.write("> A low p-value (e.g., < 0.05) suggests the variable's categories are not uniformly distributed.\n\n")
            
            # Bivariate Analysis
            if self.target_col and self.results["bivariate"]:
                f.write(f"## 5. Bivariate Analysis vs. Target Variable ('{self.target_col}')\n\n")
                
                f.write("### Continuous Features vs. Target\n\n")
                for result in self.results["bivariate"]:
                    if result['type'] == 'continuous':
                        col = result['col']
                        test = result['test']
                        f.write(f"#### Analysis of '{col}' vs. '{self.target_col}'\n")
                        f.write(f"![Boxplot of {col} vs. {self.target_col}]({result['plot_path']})\n\n")
                        
                        f.write(f"**{test['name']} Results:**\n")
                        f.write(f"- Statistic: {test['stat']:.2f}\n")
                        f.write(f"- P-value: {test['p']:.3f}\n")
                        f.write(f"> A low p-value suggests a significant difference in the mean of '{col}' across the different classes of '{self.target_col}'.\n\n")
                
                f.write("### Categorical Features vs. Target\n\n")
                for result in self.results["bivariate"]:
                    if result['type'] == 'categorical':
                        col = result['col']
                        test = result['test']
                        f.write(f"#### Analysis of '{col}' vs. '{self.target_col}'\n")
                        f.write(f"**Crosstabulation:**\n")
                        f.write(f"{result['crosstab']}\n\n")
                        
                        f.write(f"**Chi-Square Test of Independence:**\n")
                        f.write(f"- Chi-Square Statistic: {test['stat']:.2f}\n")
                        f.write(f"- P-value: {test['p']:.3f}\n")
                        f.write(f"- Cramer's V: {test['cramers_v']:.2f}\n")
                        f.write(f"> A low p-value suggests a significant association between '{col}' and '{self.target_col}'. Cramer's V measures the strength of this association (0 to 1).\n\n")

def test_eda_analyzer():
    """Standalone test function for EDAAnalyzer"""
    # Test data
    data = {
        'age': [25, 30, 45, 35, 50, 22, 60, 28, 40],
        'salary': [50000, 60000, 80000, 75000, 90000, 48000, 120000, 52000, 85000],
        'city': ['New York', 'London', 'Tokyo', 'London', 'New York', 'Tokyo', 'New York', 'Tokyo', 'New York'],
        'education_level': ['Bachelor', 'Master', 'PhD', 'Master', 'Bachelor', 'Bachelor', 'PhD', 'Master', 'PhD'],
        'subscribed': ['no', 'yes', 'yes', 'yes', 'no', 'no', 'yes', 'no', 'yes']
    }
    
    df = pd.DataFrame(data)
    schema = {
        'age': 'numerical',
        'salary': 'numerical',
        'city': 'categorical',
        'education_level': 'categorical',
        'subscribed': 'categorical'
    }
    
    print("Running EDAAnalyzer test...")
    print(f"Data shape: {df.shape}")
    
    # Test with direct parameters
    analyzer = EDAAnalyzer(
        df=df,
        schema=schema,
        target_col='subscribed',
        ordinal_cols=['education_level'],
        generate_plots=True,
        output_format='md',
        plot_style='seaborn',
        openai_api_key=None
    )
    
    analyzer.run_analysis()
    
    print("Test completed successfully!")
    print(f"Report generated at: {analyzer.report_path}")

if __name__ == '__main__':
    test_eda_analyzer()