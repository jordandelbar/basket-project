"""Generate a dataset of baskets."""
import csv
import random

import numpy as np
import pandas as pd
from loguru import logger


class BasketGenerator:
    """Basket generator class."""

    def __init__(self):
        """Basket generator class."""
        self.items_df = pd.read_csv("items.csv")
        self.items_price_range = self._read_csv("items_prices_range.csv")
        self.first_deparment = self._read_csv("first_deparment_probability.csv")
        self.markov = self._read_csv("markov_chain.csv")
        self.items_list = self._return_items_list()
        self.departments_to_items = self._get_items_to_departments()
        self.items_to_departments = self._get_departments_to_items()
        self.price_dict = self._assign_price_to_item()

    def _read_csv(self, filename: str) -> dict:
        """Read the markov chain.

        Args:
            filename (str): filename of the markov chain

        Returns:
            dict: dictionary of the markov chain
        """
        data_dict = {}

        with open(filename, "r") as file:
            reader = csv.reader(file)
            _ = next(reader)  # Read the header row
            for row in reader:
                department = row[0]
                values = [float(value) for value in row[1:]]
                data_dict[department] = values

        return data_dict

    def _get_items_to_departments(self) -> dict:
        """Collect items in departments.

        Returns:
            dict: dictionary of departments and their items
        """
        departments_items = {}

        for column in self.items_df.columns:
            departments_items[column] = self.items_df[column].tolist()

        return departments_items

    def _get_departments_to_items(self) -> dict:
        """Collect departments in items.

        Returns:
            dict: dictionary of items and their departments
        """
        return {item: department for department, items in self.departments_to_items.items() for item in items}

    def _return_items_list(self) -> list:
        """Return a list of items.

        Returns:
            list: list of items
        """
        return self.items_df.values.flatten("F").tolist()

    def _return_basket_size(self, mu: int = 8, sigma: int = 5) -> int:
        """Return the size of the basket.

        Args:
            mu (int, optional): mean of the basket size. Defaults to 8.
            sigma (int, optional): standard deviation of the basket size. Defaults to 5.

        Returns:
            int: size of the basket
        """
        random_number = int(np.random.normal(mu, sigma))
        return random_number if random_number > 0 else 1

    def _assign_price_to_item(self) -> dict:
        """Assign a price to each item.

        Returns:
            dict: dictionary of items and their price
        """
        price_range_department = {}
        price_dict = {}

        for department in self.items_price_range:
            price_range_department[department] = np.linspace(
                self.items_price_range[department][0],
                self.items_price_range[department][1],
            )

        for item in self.items_list:
            department = self.items_to_departments[item]
            price = round(np.random.choice(price_range_department[department]), 2)
            price_dict[item] = price

        return price_dict

    def transform_index_basket(self, basket: list) -> list:
        """Transform the index of a basket.

        Args:
            basket (list): basket of items

        Returns:
            list: transformed basket
        """
        return [self.items_list.index(item_ix) for item_ix in basket]

    def return_basket_price(self, basket: list) -> float:
        """Return the price of a basket.

        Args:
            basket (list): basket of items

        Returns:
            float: price of the basket
        """
        return sum([self.price_dict[item] for item in basket])

    def generate_customer_basket(self) -> tuple:
        """Generate a customer basket.

        Returns:
            list: customer basket
        """
        # 1. Define our basket
        basket = []
        # 2. Choose the number of items in the basket
        number_of_items = self._return_basket_size()
        # 3. Choose a first department based on the probability of each department
        first_department = np.random.choice(
            list(self.first_deparment.keys()),
            p=[i[0] for i in self.first_deparment.values()],
        )
        # 4. Choose an item from the chosen department
        first_item = np.random.choice(self.departments_to_items[first_department])
        basket.append(first_item)
        # 5. Choose or next department based on the probability of each department
        previous_department = first_department
        for _ in range(number_of_items - 1):
            next_department = np.random.choice(list(self.markov.keys()), p=self.markov[previous_department])
            # 6. Choose an item from the chosen department
            next_item = np.random.choice(self.departments_to_items[next_department])
            # 7. Append the item to the basket
            basket.append(next_item)
            # 8. Update the previous department
            previous_department = next_department

        return basket


def label_basket(basket: list, target_item: list) -> int:
    """Label a basket.

    Args:
        basket (list): basket of items
        target_item (list): target item

    Returns:
        int: 1 if the target item is in the basket, 0 otherwise
    """
    if any(item in basket for item in target_item):
        return 1
    else:
        return 0


if __name__ == "__main__":
    # We set the random seed
    random_state = 42
    np.random.seed(random_state)
    random.seed(random_state)

    # We instantiate our basket class
    logger.info("Instantiating basket generator class...")
    basket_generator = BasketGenerator()

    # We can generate a dataset of 100000 baskets
    logger.info("Generating dataset...")
    items = [basket_generator.generate_customer_basket() for _ in range(200000)]
    items_ix = [basket_generator.transform_index_basket(basket) for basket in items]

    logger.info("Computing labels...")
    list_items = basket_generator.departments_to_items["Meat/Seafood"]
    target_item = list_items[: len(list_items) // 12]
    labels = [label_basket(basket, target_item=target_item) for basket in items]

    # We can create our full dataset with the number of items and the price of the basket
    logger.info("Computing dataset...")
    temp_dataset = [
        [
            len(basket),
            basket_generator.return_basket_price(basket=basket),
        ]
        for basket in items
    ]

    full_dataset = pd.DataFrame(temp_dataset, columns=["number_of_items", "price_of_basket"])
    full_dataset["items"] = items
    full_dataset["items_ix"] = items_ix
    full_dataset["label"] = labels

    full_dataset.to_json("dataset.json", orient="records")
    pd.DataFrame(basket_generator.items_to_departments, index=[0]).to_json("item_departments.json", orient="records")
