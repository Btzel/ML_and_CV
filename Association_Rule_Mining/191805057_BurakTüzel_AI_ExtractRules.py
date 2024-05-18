from collections import defaultdict

# Define a function to generate association rules
def generate_rules(frequent_itemsets, item_counts, total_transactions, min_confidence):
    rules = []
    for support, itemset in frequent_itemsets:
        if len(itemset) > 1:
            for item in itemset:
                antecedent = itemset - frozenset([item])
                consequent = frozenset([item])
                
                antecedent_support = item_counts[tuple(antecedent)[0]] / total_transactions
                consequent_support = item_counts[item] / total_transactions
                
                confidence = support / antecedent_support
                lift = confidence / consequent_support
                leverage = support - antecedent_support * consequent_support
                
                if confidence == 1.0:
                    conviction = float('inf')
                else:
                    conviction = (1 - consequent_support) / (1 - confidence)
                
                zhangs_metric = (support - antecedent_support * consequent_support) / max(support * (1 - antecedent_support),
                                                                                          antecedent_support * (1 - consequent_support))
                
                if confidence >= min_confidence:
                    rules.append({
                        "antecedent": antecedent,
                        "consequent": consequent,
                        "antecedent_support": antecedent_support,
                        "consequent_support": consequent_support,
                        "support": support,
                        "confidence": confidence,
                        "lift": lift,
                        "leverage": leverage,
                        "conviction": conviction,
                        "zhangs_metric": zhangs_metric
                    })
    return rules

# My dataset
data = [
    ['Refund: Yes', 'Marital Status: Single', 'Taxable Income: 125k', 'Cheat: No'],
    ['Refund: No', 'Marital Status: Married', 'Taxable Income: 100k', 'Cheat: No'],
    ['Refund: No', 'Marital Status: Single', 'Taxable Income: 70k', 'Cheat: No'],
    ['Refund: Yes', 'Marital Status: Married', 'Taxable Income: 120k', 'Cheat: No'],
    ['Refund: No', 'Marital Status: Divorced', 'Taxable Income: 95k', 'Cheat: Yes'],
    ['Refund: No', 'Marital Status: Married', 'Taxable Income: 60k', 'Cheat: No'],
    ['Refund: Yes', 'Marital Status: Divorced', 'Taxable Income: 220k', 'Cheat: No'],
    ['Refund: No', 'Marital Status: Single', 'Taxable Income: 85k', 'Cheat: Yes'],
    ['Refund: No', 'Marital Status: Married', 'Taxable Income: 75k', 'Cheat: No'],
    ['Refund: No', 'Marital Status: Single', 'Taxable Income: 90k', 'Cheat: Yes']
]

# Step 1: Counting item occurrences
item_counts = defaultdict(int)
itemset_counts = defaultdict(int)

for transaction in data:
    for item in transaction:
        item_counts[item] += 1
    # Generate all possible combinations of items for each transaction
    for i in range(len(transaction)):
        for j in range(i+1, len(transaction)):
            itemset = frozenset([transaction[i], transaction[j]])
            itemset_counts[itemset] += 1

# Step 2: Calculating support
total_transactions = len(data)
min_support = 0.2  # Set your minimum support threshold here

frequent_itemsets_manual = []

for item, count in item_counts.items():
    support = count / total_transactions
    if support >= min_support:
        frequent_itemsets_manual.append((support, frozenset([item])))

for itemset, count in itemset_counts.items():
    support = count / total_transactions
    if support >= min_support:
        frequent_itemsets_manual.append((support, itemset))

# Step 3: Filtering frequent itemsets
frequent_itemsets_manual.sort(reverse=True)  # Sort frequent itemsets by support

# Print frequent itemsets
for support, itemset in frequent_itemsets_manual:
    print(f"Support: {support:.2f}, Itemset: {itemset}")

# Step 4: Generate association rules
min_confidence = 0.7  # Set your minimum confidence threshold here
my_rules = generate_rules(frequent_itemsets_manual, item_counts, total_transactions, min_confidence)

# Print association rules
for rule in my_rules:
    print("Antecedent:", rule["antecedent"])
    print("Consequent:", rule["consequent"])
    print("Antecedent Support:", rule["antecedent_support"])
    print("Consequent Support:", rule["consequent_support"])
    print("Support:", rule["support"])
    print("Confidence:", rule["confidence"])
    print("Lift:", rule["lift"])
    print("Leverage:", rule["leverage"])
    print("Conviction:", rule["conviction"])
    print("Zhang's Metric:", rule["zhangs_metric"])
    print("\n")

