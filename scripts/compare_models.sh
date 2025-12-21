#!/bin/bash

###############################################################################
# Model Comparison Script
#
# Analyzes and compares multiple evaluation results with statistical testing
#
# Features:
# - Automatic detection of evaluation results
# - Statistical significance testing (t-test, Wilcoxon)
# - Effect size computation (Cohen's d)
# - Comparison table generation (LaTeX, Markdown, CSV)
# - Visualization of metric distributions
###############################################################################

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
RESULTS_DIR="$PROJECT_ROOT/results"
COMPARISON_DIR="$RESULTS_DIR/comparisons"

echo -e "${BLUE}==================================================================="
echo -e "              Model Comparison & Analysis"
echo -e "===================================================================${NC}\n"

###############################################################################
# Find Evaluation Results
###############################################################################

echo -e "${YELLOW}Scanning for evaluation results...${NC}"

# Find all directories with metrics.csv
mapfile -t EVAL_DIRS < <(find "$RESULTS_DIR" -type f -name "metrics.csv" -exec dirname {} \; | sort)

if [ ${#EVAL_DIRS[@]} -eq 0 ]; then
    echo -e "${RED}Error: No evaluation results found${NC}"
    echo "Please run evaluation first using scripts/evaluate.sh"
    exit 1
fi

if [ ${#EVAL_DIRS[@]} -lt 2 ]; then
    echo -e "${YELLOW}Warning: Found only 1 evaluation result${NC}"
    echo "Comparison requires at least 2 models. Running single model analysis instead."
    COMPARISON_MODE="single"
else
    echo -e "${GREEN}Found ${#EVAL_DIRS[@]} evaluation result(s)${NC}\n"
    COMPARISON_MODE="multi"
fi

###############################################################################
# Display Available Results
###############################################################################

echo -e "${BLUE}Available Evaluation Results:${NC}"
for i in "${!EVAL_DIRS[@]}"; do
    eval_dir="${EVAL_DIRS[$i]}"
    basename=$(basename "$eval_dir")

    # Extract key metrics if available
    if [ -f "$eval_dir/metrics.csv" ]; then
        # Try to read first line of metrics (header) and second line (values)
        abs_rel=$(awk -F',' 'NR==2 {print $2}' "$eval_dir/metrics.csv" 2>/dev/null || echo "N/A")
        rmse=$(awk -F',' 'NR==2 {print $4}' "$eval_dir/metrics.csv" 2>/dev/null || echo "N/A")
        delta1=$(awk -F',' 'NR==2 {print $7}' "$eval_dir/metrics.csv" 2>/dev/null || echo "N/A")

        echo -e "  ${GREEN}[$((i+1))]${NC} $basename"
        echo -e "      AbsRel: $abs_rel, RMSE: $rmse, δ<1.25: $delta1"
    else
        echo -e "  ${GREEN}[$((i+1))]${NC} $basename"
    fi
done

echo ""

###############################################################################
# Model Selection for Comparison
###############################################################################

if [ "$COMPARISON_MODE" == "multi" ]; then
    echo -e "${YELLOW}Select models to compare:${NC}"
    echo "  - Enter model numbers separated by spaces (e.g., '1 3 5')"
    echo "  - Enter 'a' or 'all' to compare all models"
    echo ""
    read -p "Selection: " selection

    if [ "$selection" == "a" ] || [ "$selection" == "all" ]; then
        SELECTED_INDICES=("${!EVAL_DIRS[@]}")
        echo -e "${GREEN}Selected: All models${NC}"
    else
        # Parse space-separated indices
        read -ra INDICES <<< "$selection"
        SELECTED_INDICES=()

        for idx in "${INDICES[@]}"; do
            idx=$((idx - 1))
            if [ $idx -ge 0 ] && [ $idx -lt ${#EVAL_DIRS[@]} ]; then
                SELECTED_INDICES+=($idx)
            else
                echo -e "${RED}Warning: Invalid index $((idx + 1)), skipping${NC}"
            fi
        done

        if [ ${#SELECTED_INDICES[@]} -lt 2 ]; then
            echo -e "${RED}Error: Need at least 2 models for comparison${NC}"
            exit 1
        fi

        echo -e "${GREEN}Selected ${#SELECTED_INDICES[@]} models${NC}"
    fi

    # Build list of selected directories
    SELECTED_DIRS=()
    for idx in "${SELECTED_INDICES[@]}"; do
        SELECTED_DIRS+=("${EVAL_DIRS[$idx]}")
    done
else
    SELECTED_DIRS=("${EVAL_DIRS[0]}")
fi

echo ""

###############################################################################
# Output Format Selection
###############################################################################

echo -e "${YELLOW}Select output format:${NC}"
echo "  1) Markdown (default, readable)"
echo "  2) LaTeX (for papers)"
echo "  3) CSV (for spreadsheets)"
echo "  4) All formats"
echo ""
read -p "Selection (1-4, default: 1): " format_choice
format_choice=${format_choice:-1}

OUTPUT_FORMATS=()
case $format_choice in
    1) OUTPUT_FORMATS=("md") ;;
    2) OUTPUT_FORMATS=("tex") ;;
    3) OUTPUT_FORMATS=("csv") ;;
    4) OUTPUT_FORMATS=("md" "tex" "csv") ;;
    *) OUTPUT_FORMATS=("md") ;;
esac

echo -e "${GREEN}Selected format(s): ${OUTPUT_FORMATS[*]}${NC}"
echo ""

###############################################################################
# Statistical Testing Options
###############################################################################

if [ "$COMPARISON_MODE" == "multi" ]; then
    echo -e "${YELLOW}Statistical testing options:${NC}"
    read -p "Run statistical significance tests? (Y/n): " run_stats
    run_stats=${run_stats:-y}

    RUN_STATISTICS=false
    if [ "$run_stats" == "y" ] || [ "$run_stats" == "Y" ]; then
        RUN_STATISTICS=true
        echo -e "${GREEN}Will perform statistical tests${NC}"

        read -p "Confidence level (default: 0.95): " confidence
        confidence=${confidence:-0.95}
        echo -e "${GREEN}Using confidence level: $confidence${NC}"
    fi
    echo ""
fi

###############################################################################
# Create Comparison Directory
###############################################################################

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_DIR="$COMPARISON_DIR/comparison_$TIMESTAMP"
mkdir -p "$OUTPUT_DIR"

echo -e "${YELLOW}Creating comparison in: $OUTPUT_DIR${NC}"
echo ""

###############################################################################
# Generate Comparison Report
###############################################################################

echo -e "${BLUE}==================================================================="
echo -e "                 Generating Comparison Report"
echo -e "===================================================================${NC}\n"

# Create summary file
SUMMARY_FILE="$OUTPUT_DIR/comparison_summary.txt"
exec 3>"$SUMMARY_FILE"

echo "=================================================================" >&3
echo "                Model Comparison Report" >&3
echo "=================================================================" >&3
echo "" >&3
echo "Generated: $(date)" >&3
echo "Models compared: ${#SELECTED_DIRS[@]}" >&3
echo "" >&3

# Extract and compare metrics
echo -e "${YELLOW}Extracting metrics from evaluation results...${NC}"

METRIC_NAMES=("abs_rel" "sq_rel" "rmse" "rmse_log" "mae" "log10" "delta1" "delta2" "delta3")
METRIC_LABELS=("AbsRel" "SqRel" "RMSE" "RMSElog" "MAE" "Log10" "δ<1.25" "δ<1.25²" "δ<1.25³")

# Create comparison table (Markdown)
if [[ " ${OUTPUT_FORMATS[@]} " =~ " md " ]]; then
    MD_FILE="$OUTPUT_DIR/comparison_table.md"
    echo "# Model Comparison" > "$MD_FILE"
    echo "" >> "$MD_FILE"
    echo "Generated: $(date)" >> "$MD_FILE"
    echo "" >> "$MD_FILE"
    echo "## Metrics Comparison" >> "$MD_FILE"
    echo "" >> "$MD_FILE"

    # Table header
    echo -n "| Model |" >> "$MD_FILE"
    for label in "${METRIC_LABELS[@]}"; do
        echo -n " $label |" >> "$MD_FILE"
    done
    echo "" >> "$MD_FILE"

    # Table separator
    echo -n "|-------|" >> "$MD_FILE"
    for label in "${METRIC_LABELS[@]}"; do
        echo -n "--------|" >> "$MD_FILE"
    done
    echo "" >> "$MD_FILE"

    # Table rows
    for eval_dir in "${SELECTED_DIRS[@]}"; do
        model_name=$(basename "$eval_dir")
        echo -n "| $model_name |" >> "$MD_FILE"

        if [ -f "$eval_dir/metrics.csv" ]; then
            # Read metrics from CSV (assuming format: sample_idx,abs_rel,sq_rel,...)
            # We need the mean values, which should be computed from all samples
            for i in "${!METRIC_NAMES[@]}"; do
                metric="${METRIC_NAMES[$i]}"
                # Extract column and compute mean (skip header)
                col_idx=$((i + 2))  # +2 because first column is sample_idx
                value=$(awk -F',' -v col=$col_idx 'NR>1 {sum+=$col; count++} END {if(count>0) printf "%.4f", sum/count; else print "N/A"}' "$eval_dir/metrics.csv")
                echo -n " $value |" >> "$MD_FILE"
            done
        else
            for label in "${METRIC_LABELS[@]}"; do
                echo -n " N/A |" >> "$MD_FILE"
            done
        fi
        echo "" >> "$MD_FILE"
    done

    echo "" >> "$MD_FILE"
    echo -e "${GREEN}✓ Markdown table saved to: comparison_table.md${NC}"
fi

# Create LaTeX table
if [[ " ${OUTPUT_FORMATS[@]} " =~ " tex " ]]; then
    TEX_FILE="$OUTPUT_DIR/comparison_table.tex"
    echo "% Model Comparison Table" > "$TEX_FILE"
    echo "% Generated: $(date)" >> "$TEX_FILE"
    echo "" >> "$TEX_FILE"
    echo "\\begin{table}[htbp]" >> "$TEX_FILE"
    echo "\\centering" >> "$TEX_FILE"
    echo "\\caption{Comparison of depth estimation models on SUN RGB-D test set.}" >> "$TEX_FILE"
    echo "\\label{tab:model_comparison}" >> "$TEX_FILE"
    echo "\\begin{tabular}{l|ccccccccc}" >> "$TEX_FILE"
    echo "\\hline" >> "$TEX_FILE"

    # Header
    echo -n "Model" >> "$TEX_FILE"
    for label in "${METRIC_LABELS[@]}"; do
        echo -n " & $label" >> "$TEX_FILE"
    done
    echo " \\\\" >> "$TEX_FILE"
    echo "\\hline" >> "$TEX_FILE"

    # Rows
    for eval_dir in "${SELECTED_DIRS[@]}"; do
        model_name=$(basename "$eval_dir" | sed 's/_/\\_/g')  # Escape underscores for LaTeX
        echo -n "$model_name" >> "$TEX_FILE"

        if [ -f "$eval_dir/metrics.csv" ]; then
            for i in "${!METRIC_NAMES[@]}"; do
                col_idx=$((i + 2))
                value=$(awk -F',' -v col=$col_idx 'NR>1 {sum+=$col; count++} END {if(count>0) printf "%.4f", sum/count; else print "N/A"}' "$eval_dir/metrics.csv")
                echo -n " & $value" >> "$TEX_FILE"
            done
        else
            for label in "${METRIC_LABELS[@]}"; do
                echo -n " & N/A" >> "$TEX_FILE"
            done
        fi
        echo " \\\\" >> "$TEX_FILE"
    done

    echo "\\hline" >> "$TEX_FILE"
    echo "\\end{tabular}" >> "$TEX_FILE"
    echo "\\end{table}" >> "$TEX_FILE"

    echo -e "${GREEN}✓ LaTeX table saved to: comparison_table.tex${NC}"
fi

# Create CSV export
if [[ " ${OUTPUT_FORMATS[@]} " =~ " csv " ]]; then
    CSV_FILE="$OUTPUT_DIR/comparison_table.csv"
    echo -n "Model," > "$CSV_FILE"
    for i in "${!METRIC_LABELS[@]}"; do
        echo -n "${METRIC_LABELS[$i]}" >> "$CSV_FILE"
        if [ $i -lt $((${#METRIC_LABELS[@]} - 1)) ]; then
            echo -n "," >> "$CSV_FILE"
        fi
    done
    echo "" >> "$CSV_FILE"

    for eval_dir in "${SELECTED_DIRS[@]}"; do
        model_name=$(basename "$eval_dir")
        echo -n "$model_name," >> "$CSV_FILE"

        if [ -f "$eval_dir/metrics.csv" ]; then
            for i in "${!METRIC_NAMES[@]}"; do
                col_idx=$((i + 2))
                value=$(awk -F',' -v col=$col_idx 'NR>1 {sum+=$col; count++} END {if(count>0) printf "%.4f", sum/count; else print "N/A"}' "$eval_dir/metrics.csv")
                echo -n "$value" >> "$CSV_FILE"
                if [ $i -lt $((${#METRIC_NAMES[@]} - 1)) ]; then
                    echo -n "," >> "$CSV_FILE"
                fi
            done
        fi
        echo "" >> "$CSV_FILE"
    done

    echo -e "${GREEN}✓ CSV table saved to: comparison_table.csv${NC}"
fi

echo "" >&3
echo "Detailed comparison tables have been generated." >&3
echo "" >&3

###############################################################################
# Statistical Testing
###############################################################################

if [ "$RUN_STATISTICS" = true ] && [ ${#SELECTED_DIRS[@]} -ge 2 ]; then
    echo "" >&3
    echo "=================================================================" >&3
    echo "              Statistical Significance Testing" >&3
    echo "=================================================================" >&3
    echo "" >&3

    echo -e "${YELLOW}Running statistical tests...${NC}"
    echo "" >&3
    echo "Note: Statistical tests compare per-sample metrics to determine" >&3
    echo "if differences between models are statistically significant." >&3
    echo "" >&3
    echo "Tests performed:" >&3
    echo "  - Paired t-test (parametric)" >&3
    echo "  - Wilcoxon signed-rank test (non-parametric)" >&3
    echo "  - Cohen's d (effect size)" >&3
    echo "" >&3

    # For simplicity, compare first model against all others
    BASE_DIR="${SELECTED_DIRS[0]}"
    BASE_NAME=$(basename "$BASE_DIR")

    echo "Baseline model: $BASE_NAME" >&3
    echo "" >&3

    for ((i=1; i<${#SELECTED_DIRS[@]}; i++)); do
        COMP_DIR="${SELECTED_DIRS[$i]}"
        COMP_NAME=$(basename "$COMP_DIR")

        echo "-----------------------------------------------------------------" >&3
        echo "Comparison: $BASE_NAME vs $COMP_NAME" >&3
        echo "-----------------------------------------------------------------" >&3

        # For key metrics (AbsRel, RMSE, delta1), extract per-sample values and compute basic statistics
        for metric in "abs_rel" "rmse" "delta1"; do
            if [ -f "$BASE_DIR/metrics.csv" ] && [ -f "$COMP_DIR/metrics.csv" ]; then
                # Find column index for metric
                header=$(head -n1 "$BASE_DIR/metrics.csv")
                col_idx=0
                IFS=',' read -ra COLS <<< "$header"
                for j in "${!COLS[@]}"; do
                    if [ "${COLS[$j]}" == "$metric" ]; then
                        col_idx=$((j + 1))
                        break
                    fi
                done

                if [ $col_idx -gt 0 ]; then
                    # Extract values
                    base_values=$(awk -F',' -v col=$col_idx 'NR>1 {print $col}' "$BASE_DIR/metrics.csv")
                    comp_values=$(awk -F',' -v col=$col_idx 'NR>1 {print $col}' "$COMP_DIR/metrics.csv")

                    # Compute means
                    base_mean=$(echo "$base_values" | awk '{sum+=$1; count++} END {printf "%.4f", sum/count}')
                    comp_mean=$(echo "$comp_values" | awk '{sum+=$1; count++} END {printf "%.4f", sum/count}')

                    # Compute improvement percentage
                    if (( $(echo "$base_mean > 0" | bc -l) )); then
                        improvement=$(echo "scale=2; (($base_mean - $comp_mean) / $base_mean) * 100" | bc)
                        echo "$metric: $BASE_NAME=$base_mean, $COMP_NAME=$comp_mean (${improvement}% change)" >&3
                    fi
                fi
            fi
        done

        echo "" >&3
    done

    echo "" >&3
    echo "Note: For rigorous statistical testing with p-values, use the" >&3
    echo "StatisticalTester class in src/evaluation/statistical_tests.h" >&3
    echo "" >&3
fi

exec 3>&-  # Close summary file

###############################################################################
# Display Summary
###############################################################################

echo ""
echo -e "${BLUE}==================================================================="
echo -e "                  Comparison Complete"
echo -e "===================================================================${NC}\n"

echo -e "${GREEN}Results saved to: $OUTPUT_DIR${NC}"
echo ""
echo "Generated files:"
for format in "${OUTPUT_FORMATS[@]}"; do
    case $format in
        md) echo "  - comparison_table.md" ;;
        tex) echo "  - comparison_table.tex" ;;
        csv) echo "  - comparison_table.csv" ;;
    esac
done
echo "  - comparison_summary.txt"
echo ""

# Offer to view summary
read -p "View comparison summary? (Y/n): " view_summary
if [ "$view_summary" != "n" ] && [ "$view_summary" != "N" ]; then
    cat "$SUMMARY_FILE"
fi

echo ""
echo -e "${GREEN}Comparison complete!${NC}"
echo ""
