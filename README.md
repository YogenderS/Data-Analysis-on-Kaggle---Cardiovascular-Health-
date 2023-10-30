# Data-Analysis-on-Kaggle---Cardiovascular-Health-
Cardiovascular illnesses (CVDs) are the major cause of death worldwide. CVDs include coronary heart disease, cerebrovascular disease, rheumatic heart disease, and other heart and blood vessel problems. According to the World Health Organization, 17.9 million people die each year. Heart attacks and strokes account for more than four out of every five CVD deaths, with one-third of these deaths occurring before the age of 70. A comprehensive database for factors that contribute to a heart attack has been constructed.

# Problem Statement
The main purpose here is to collect characteristics of Heart Attack or factors that contribute to it. The size of the dataset is 1319 samples, which have nine fields, where eight fields are for input fields and one field for an output field. Age, gender(0 for Female, 1 for Male) ,heart rate (impulse), systolic BP (pressurehight), diastolic BP (pressurelow), blood sugar(glucose), CK-MB (kcm), and Test-Troponin (troponin) are representing the input fields, while the output field pertains to the presence of heart attack (class), which is divided into two categories (negative and positive); negative refers to the absence of a heart attack, while positive refers to the presence of a heart attack.

🛠 Tools Used
Tools used: Python (library: Seaborn, Matplotlib, Pandas, and Numpy)

MySQL

Excel

Power BI

Documentation
Python Analysis:-
Table of Contents:
1️⃣ Load Dataset

2️⃣ Import Libraries

3️⃣ Exploratory Data Analysis

4️⃣ Checking Null Values and Cleaning Data

5️⃣ Visualization :

Sales Analysis:
1️⃣ Analyze the number of products sold and revenue generated to understand sales performance over time.

2️⃣ Identify customer demographics to determine which groups are purchasing the most products.

3️⃣ Track availability and stock levels to ensure the right products are in stock when customers are ready to buy.

Operational Analysis:
1️⃣ Analyze lead times, order quantities, and production volumes to optimize inventory management and reduce stockouts.

2️⃣ Track manufacturing lead times and costs to identify areas for improvement and cost savings.

3️⃣ Monitor inspection results and defect rates to identify quality issues and improve manufacturing processes.

Shipping Analysis:
1️⃣ Analyze costs, transportation modes, and routes to optimize logistics and reduce shipping costs.

2️⃣ Monitor shipping times, shipping carriers, and modes of transportation to ensure timely delivery to customers.

3️⃣ Track shipping costs associated with shipping carriers and revenue generated to identify areas for cost savings.

Libraries Used for the Analysis:
Data Manipulation: NumPy, Pandas

Data Visualization: Seaborn, Matplotlib

Mysql Analysis:-
I used MySQL to analyze supply chain data and answer specific questions.

📊 Total Revenue by Product Type: Calculate the total revenue generated by each product type. This can help identify the most profitable products.

🚚 Shipping Costs by Carrier: Calculate the average shipping costs for each shipping carrier to evaluate which carrier is the most cost-effective.

⏳ Lead Times by Supplier: Calculate average lead times for each supplier to assess their reliability in delivering products on time.

🌍 Total Products Sold by Location: Determine the total number of products sold in each location to understand regional demand.

🔍 Defect Rates by Inspection Results: Calculate defect rates for different inspection results to assess product quality.

💰 Ranking Products by Profitability: Write a query to rank products based on profitability, where profitability is calculated as (revenue generated - manufacturing costs - shipping costs).

📦 Supplier Performance Analysis: Calculate the average lead time for each supplier and identify suppliers with lead times higher than the overall average.

🏆 Top 5 Shipping Carriers by Cost Efficiency: Determine the top 5 shipping carriers with the lowest average shipping costs per product shipped.

Excel Analysis:-
Excel

Excel Analysis Summary for Supply Chain Capstone Project

Create an Excel dashboard for your supply chain capstone project and visualize key insights from your supply chain dataset.

📊 Visualizing Key Insights

From the overall analysis, I have discovered some key insights:

💰 Skincare products are the most profitable, generating significant revenue. Haircare products come next in popularity, followed by cosmetics.

🚚 Carrier C has the highest shipping costs on average, with carrier A as the second-costliest option. Carrier B, on the other hand, is the most affordable choice.

🛳️ Among the four transportation modes (air, rail, road, and sea), air transportation is associated with higher shipping costs, while sea transportation offers a more economical solution.

📈 The top locations for product sales are Kolkata and Delhi, closely followed by Mumbai and Chennai, each with a substantial number of products sold. In contrast, Bangalore reports the lowest number of products sold.

❌ Among the three types of products, haircare exhibits the highest defect rate at 37%, followed by skincare and cosmetics.

💼 The manufacturing cost of skincare products is higher than that of haircare and cosmetics.

This Excel dashboard provides a concise overview of the supply chain dataset, making it easier to understand and make data-driven decisions.

Power BI:-
Power BI - Dashboard

In Powe Bi, build charts, graphs, and KPIs to visualize the data and understand it more interactively, along with that, build some slicers to make it more user-friendly to customize the charts and graphs.

Build some cards to understand some key performance indicators, including:

Key Performance Indicators (KPIs):
1️⃣ Total Revenue: 577,000

2️⃣ Average Product Price: 49.46

3️⃣ Total Products Sold: 46,000

4️⃣ Manufacturing Cost: 47.27

5️⃣ Total Available Products: 4,840

6️⃣ Total Stock Value: 4,777

Charts and Graphs:
Average Defect Rate, Manufacturing Lead Time, and Manufacturing Cost by Product

Customer Demographics vs. Number of Products Sold by Product Type

Total Revenue Generated by Product Type and Total Products Sold by Product Type

Average Manufacturing Cost vs. Average Price by Product Type

Transportation Modes vs. Routes with Average Cost

Defect Rate vs. Inspection Rate by Product Type

Transportation Modes and Shipping Carriers with Average Shipping Time

Interactive Slicers:
Product Type

Transportation Mode

Shipping Carrier

Routes

Suppliers' Name

Insights
The highest number of products sold among the three product categories is skincare, which accounts for 45% of the business. Haircare follows at 29%, and cosmetics contribute 25% to the revenue.

Analyzing customer demographics versus the number of products sold by product type reveals that the female group purchases higher-quality skincare and cosmetic products, while the male group opts for products of roughly equal quality in terms of haircare and cosmetics.

An unknown group category purchases a higher quantity of all three product types. Skincare products are the most popular among all four product categories, indicating a high demand.

Analyzing product availability and stock levels shows that the company maintains an equal quantity of inventory for haircare and skincare products, with slightly less stock for cosmetics. Skincare products exhibit higher availability and lower stock levels, enabling quick manufacturing and shipment.
In contrast, cosmetics and haircare have higher stock levels and lower availability, meaning the company faces challenges in promptly shipping these products due to longer manufacturing times.

Skincare products have higher order quantities and longer lead times, often associated with higher production volumes.
Higher production volumes necessitate longer lead times to ensure adequate time for manufacturing products to meet customer demand.

Haircare products also exhibit longer lead times and higher production volumes, possibly due to the need for more specialized ingredients or manufacturing processes.

All product categories show a higher defect rate.

An analysis of revenue generated by shipping carriers indicates that shipping carrier B is costlier but generates higher revenue.

According to the graph, the fastest and most efficient shipping option is Carrier B, and the most efficient transportation mode is road in all four transportation modes.

Links
Power Bi Dashbaord
