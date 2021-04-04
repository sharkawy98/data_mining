# %%
# Import pandas library and read the dataset
import pandas as pd
coffee_shop = pd.read_excel('data/coffe_shop.xlsx')
print(coffee_shop.head())


# %%
# Take user input for support count and confidence interval
min_support = float(input('Enter your support count: '))
min_confidence = float(input('Enter your confidence interval in the range [0, 1]: '))


# %%
# it will concatenate transaction items without duplicates (using sets)
# this will make algorithm searching easier 
def concat_items(row):
    row['Items'] = set((row['Item 1'], row['Item 2'], row['Item 3']))
    return row


# %%
# Concatenate the whole three items in 1 cell for all rows
coffee_shop = coffee_shop.apply(concat_items, axis=1)

# Final DataFrame with "transaction number" and "items" only 
coffee_shop_final = coffee_shop[['Transaction Number', 'Items']]
print(coffee_shop_final.head())


# %%
# Get all unique transaction's items
unique_items = set.union(set(coffee_shop['Item 1']), set(coffee_shop['Item 2']), set(coffee_shop['Item 3']))
print(unique_items)


# %%
# Build first level item sets
sorted_items = [set(item.split(',')) for item in sorted(unique_items)]
level_1 = pd.DataFrame({'item_sets': sorted_items})
print(level_1)


# %%
def get_item_set_support(item_set):
    return coffee_shop_final[coffee_shop_final['Items'].apply(lambda items: item_set.issubset(items))].count()[0]


# %%
# Get support to first level item sets
level_1['support'] = level_1['item_sets'].apply(get_item_set_support)
level_1 = level_1[level_1.support >= min_support].reset_index(drop=True)


# %%
# Python dictionary to hold all levels' dataframes 
# this will be used at asscoiation rules calculations
levels_dfs = {'level_1_df': level_1}


# %%
# Join an item sets level to itself, to get the next level itemsets
def join_previous_level(level_num):
    prev_level = levels_dfs['level_' +str(level_num-1)+ '_df']
    item_sets = []

    for idx, row in prev_level.iterrows():
        for i in range(idx+1, prev_level.shape[0]):
            union = set(prev_level.at[idx, 'item_sets'].union(prev_level.at[i, 'item_sets']))
            if union not in item_sets:
                item_sets.append(union)

    if item_sets is None:
        return None
        
    return pd.DataFrame({'item_sets': item_sets})


# %%
# Iterate over to get the next levels itemsets
# and stop over if no more levels to produce 
# or the minimum support count hasn't met
for i in [2,3]:
    curr_level = join_previous_level(i)
    if curr_level is None:
        break
    
    curr_level['support'] = curr_level.item_sets.map(get_item_set_support)
    curr_level = curr_level[curr_level.support >= min_support].reset_index(drop=True)
    if curr_level.shape[0] == 0: 
        break

    # add the new level to the levels dictionary
    levels_dfs['level_' +str(i)+ '_df'] = curr_level


# %%
# Get the most frequent itemset and its level 
most_freq_itemsets = None
most_freq_level = 0
for df in levels_dfs.values():
    most_freq_itemsets = df
    most_freq_level += 1
    print(df.sort_values(by='support', ascending=False).to_string())


# %%
print('The most frequent item sets (sorted by highest support) are: ')
print('-'*60)
print(most_freq_itemsets.sort_values(by='support', ascending=False).to_string())


# %%
# Stop execution if the most frequent itemset level is one
if most_freq_level == 1:
    print('-'*40)
    raise SystemExit('No association rules for the first level itemsets')


# %%
# Return a DataFrame contains all the association rules for an itemset
import itertools 
def get_itemset_associations(item_set):
    ifs = []
    for n in range(len(item_set)-1, 0, -1):
        ifs += [set(i) for i in itertools.combinations(item_set, n)]

    thens = []
    for s in ifs:
        thens += [item_set - s]

    return pd.DataFrame({'if': ifs, 'then': thens})


# %%
# Get all association rules for each most frequent itemset
associations = most_freq_itemsets.item_sets.map(get_itemset_associations)


# %%
# Combine all that associations in one dataframe
# to calculate their confidence and display final results
conc_assoc = pd.concat([a for a in associations], ignore_index=True)
conc_assoc.head()


# %%
# Add "confidence" cell to each row of a dataset
def get_confidence(row):
    # The rule:
      # f={1, 2, 3} and g={1, 2}
      # g => f - g  "{1,2} => {3}"
      # confidence = f_support / g_support
    g = row['if']
    f = row['if'].union(row['then'])

    g_level = levels_dfs['level_' +str(len(g))+ '_df']
    f_level = levels_dfs['level_' +str(len(f))+ '_df']

    g_support = g_level.loc[g_level[g_level['item_sets'] == g].index.values[0], 'support']
    f_support = f_level.loc[f_level[f_level['item_sets'] == f].index.values[0], 'support']

    row['confidence'] = round(f_support / g_support, 2)
    return row


# %%
# Get confidence for each association rule "if => then"
conc_assoc = conc_assoc.apply(get_confidence, axis=1)
print('The final association rules with thier confidence (sorted by highest confidence):')
print('-'*80)
print(conc_assoc[conc_assoc.confidence >= min_confidence].sort_values(by='confidence', ascending=False).to_string())
