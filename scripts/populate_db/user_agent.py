import os
import random
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from tqdm import tqdm

sys.path.append("../../src")
from text2sql.data.datasets import MysqlDataset

from agent_options import (FEEDBACK_OPTIONS, NEGATIVE_COMMENTS_DELIVERY,
                           NEGATIVE_COMMENTS_FOOD, NOTIFICATIONS,
                           PAYMENT_METHODS, POSITIVE_COMMENTS_DELIVERY,
                           POSITIVE_COMMENTS_FOOD, REFUND_REASONS)
from agent_utils import (DatabaseConnection, get_random_row, get_related_rows,
                         random_date_after_date, random_selection)
from utils import Config, insert_data

load_dotenv()

JAN_FIRST_2023 = datetime(2023, 1, 1)
LAST_ORDER_DATE = JAN_FIRST_2023
LAST_REFUND_DATE = JAN_FIRST_2023
LAST_REVIEW_DATE = JAN_FIRST_2023
LAST_DELIVERY_FEEDBACK = JAN_FIRST_2023


ORDER_TIMEDELTA_RANGE = timedelta(minutes=6)
REFUND_TIMEDELTA_RANGE = timedelta(hours=4)
REVIEW_TIMEDELTA_RANGE = timedelta(hours=4)
DELIVERY_FEEDBACK_TIMEDELTA_RANGE = timedelta(hours=4)


LAST_SCHEDULED_ORDER = JAN_FIRST_2023
LAST_APP_REVIEW = JAN_FIRST_2023
LAST_SCHEDULED_ORDER = JAN_FIRST_2023
LAST_NOTIFICATION_DATE = JAN_FIRST_2023
LAST_SUPPORT_TICKET_DATE = JAN_FIRST_2023
UTILITY_TIMEDELTA_RANGE = timedelta(hours=10)


@dataclass
class UserData:
    user_id: int
    address: str


class OrderManager:
    def __init__(self, db: DatabaseConnection, user_data: UserData):
        self.db = db
        self.user_data = user_data

    def create_order(self) -> int:
        global LAST_ORDER_DATE
        if LAST_ORDER_DATE > datetime.now():
            return
        """Create a new order with random menu items from a random restaurant."""
        restaurant = self.db.get_random_row("Restaurants", columns=["restaurant_id"])
        menu_items = self.db.get_related_rows(
            "MenuItems", "restaurant_id", restaurant["restaurant_id"]
        )
        selected_items = random_selection(menu_items)

        total_price = Decimal(sum(Decimal(item["price"]) for item in selected_items))

        order_date = random_date_after_date(LAST_ORDER_DATE, ORDER_TIMEDELTA_RANGE)

        order_data = [
            {
                "user_id": self.user_data.user_id,
                "restaurant_id": restaurant["restaurant_id"],
                "total_price": float(total_price),  # Convert Decimal to float for DB
                "order_status": "Completed",
                "order_date": order_date,
            }
        ]

        [order_id] = self.db.insert("Orders", order_data)
        LAST_ORDER_DATE = order_date
        self._create_order_items(order_id, selected_items)
        self._create_payment(order_id, order_date)
        self._create_delivery(order_id, order_date)

        return order_id

    def _create_order_items(
        self, order_id: int, menu_items: List[Dict[str, Any]]
    ) -> None:
        """Create order items for a given order."""
        order_items: Dict[int, Dict[str, Any]] = {}

        for item in menu_items:
            menu_item_id = item["menu_item_id"]
            if menu_item_id not in order_items:
                order_items[menu_item_id] = {
                    "order_id": order_id,
                    "menu_item_id": menu_item_id,
                    "quantity": 1,
                    "price": float(item["price"]),
                }
            else:
                order_items[menu_item_id]["quantity"] += 1
                order_items[menu_item_id]["price"] += float(item["price"])

        self.db.insert("OrderItems", list(order_items.values()))

    def _create_payment(self, order_id: int, order_date: datetime) -> None:
        """Create a payment record for an order."""
        payment_data = [
            {
                "order_id": order_id,
                "payment_method": random.choice(PAYMENT_METHODS),
                "payment_status": "Completed",
                "payment_date": order_date,
            }
        ]
        self.db.insert("Payments", payment_data)

    def _create_delivery(self, order_id: int, order_date: datetime) -> None:
        """Create a delivery record for an order."""
        delivery_person = self.db.get_random_row(
            "DeliveryPersons", columns=["delivery_person_id"]
        )

        delivery_data = [
            {
                "order_id": order_id,
                "delivery_person_id": delivery_person["delivery_person_id"],
                "delivery_status": "Delivered",
                "delivery_address": self.user_data.address,
                "delivery_date": order_date,
            }
        ]
        self.db.insert("Deliveries", delivery_data)


class User:
    def __init__(self, dataset: MysqlDataset, db_name: str, **user_data):
        self.data = UserData(**user_data)
        self.db = DatabaseConnection(dataset, db_name)
        self.order_manager = OrderManager(self.db, self.data)

    def _get_order(
        self, order_id: Optional[int], required_columns: List[str]
    ) -> Optional[Dict[str, Any]]:
        """Get a specific or random order for the user."""
        if not order_id:
            return self.db.get_random_row(
                "Orders", "user_id", self.data.user_id, required_columns
            )

        orders = self.db.get_related_rows(
            "Orders", "order_id", order_id, required_columns
        )
        return orders[0] if orders else None

    def make_order(self) -> int:
        """Create a new order for the user."""
        return self.order_manager.create_order()

    def request_refund(
        self, order_id: Optional[int] = None
    ) -> Optional[Dict[str, Any]]:
        global LAST_REFUND_DATE
        if LAST_REFUND_DATE > datetime.now():
            return
        """Request a refund for a specific or random completed order."""
        order = self._get_order(order_id, ["order_id", "total_price", "order_date"])
        if not order:
            return None

        refund_amount = round(float(order["total_price"]) * random.uniform(0.5, 1.0), 2)
        refund_date = random_date_after_date(
            max(order["order_date"], LAST_REFUND_DATE),
            timedelta_range=REFUND_TIMEDELTA_RANGE,
            exp=True,
        )

        refund_data = [
            {
                "order_id": order["order_id"],
                "refund_reason": random.choice(REFUND_REASONS),
                "refund_amount": refund_amount,
                "refund_status": "Approved" if random.random() < 0.92 else "Rejected",
                "requested_at": refund_date.strftime("%Y-%m-%d %H:%M:%S"),
            }
        ]

        self.db.insert("Refunds", refund_data)
        LAST_REFUND_DATE = refund_date
        return refund_data[0]

    def write_review(self, order_id: Optional[int] = None) -> Optional[Dict[str, Any]]:
        """Write a review for a specific or random completed order."""
        global LAST_REVIEW_DATE
        if LAST_REVIEW_DATE > datetime.now():
            return
        order = self._get_order(order_id, ["order_id", "restaurant_id", "order_date"])
        if not order:
            return None

        rating = random.randint(1, 5)
        comments = POSITIVE_COMMENTS_FOOD if rating >= 4 else NEGATIVE_COMMENTS_FOOD
        review_date = random_date_after_date(
            max(order["order_date"], LAST_REVIEW_DATE),
            timedelta_range=REVIEW_TIMEDELTA_RANGE,
            exp=True,
        )

        review_data = [
            {
                "user_id": self.data.user_id,
                "restaurant_id": order["restaurant_id"],
                "rating": rating,
                "comments": random.choice(comments),
                "created_at": review_date,
            }
        ]

        self.db.insert("Reviews", review_data)
        LAST_REVIEW_DATE = review_date
        return review_data[0]

    def add_favorite(self) -> Optional[Dict[str, Any]]:
        """Add a random restaurant from previous orders to favorites."""
        orders = self.db.get_related_rows(
            "Orders", "user_id", self.data.user_id, ["restaurant_id"]
        )
        if not orders:
            return None

        existing_favorites = self.db.get_related_rows(
            "Favorites", "user_id", self.data.user_id, ["restaurant_id"]
        )
        existing_restaurant_ids = {f["restaurant_id"] for f in existing_favorites}

        new_restaurants = [
            order
            for order in orders
            if order["restaurant_id"] not in existing_restaurant_ids
        ]
        if not new_restaurants:
            return None

        selected_restaurant = random.choice(new_restaurants)
        favorite_data = [
            {
                "user_id": self.data.user_id,
                "restaurant_id": selected_restaurant["restaurant_id"],
                "created_at": random_date_after_date(JAN_FIRST_2023),
            }
        ]

        self.db.insert("Favorites", favorite_data)
        return favorite_data[0]

    def leave_delivery_feedback(self, order_id: Optional[int] = None) -> Dict[str, Any]:
        """Leave feedback for a specific or random delivery."""
        global LAST_DELIVERY_FEEDBACK
        if LAST_DELIVERY_FEEDBACK > datetime.now():
            return
        if not order_id:
            order = self._get_order(None, ["order_id"])
            delivery = self.db.get_random_row(
                "Deliveries",
                "order_id",
                order["order_id"],
                ["delivery_id", "delivery_date"],
            )
        else:
            delivery = self.db.get_related_rows(
                "Deliveries", "order_id", order_id, ["delivery_date", "delivery_id"]
            )[0]
        delivery_id = delivery["delivery_id"]

        rating = random.randint(1, 5)
        comments = (
            POSITIVE_COMMENTS_DELIVERY if rating >= 4 else NEGATIVE_COMMENTS_DELIVERY
        )
        feedback_date = random_date_after_date(
            max(delivery["delivery_date"], LAST_DELIVERY_FEEDBACK),
            timedelta_range=DELIVERY_FEEDBACK_TIMEDELTA_RANGE,
            exp=True,
        )

        feedback_data = [
            {
                "delivery_id": delivery_id,
                "user_id": self.data.user_id,
                "rating": rating,
                "comments": random.choice(comments),
                "created_at": feedback_date,
            }
        ]

        self.db.insert("DeliveryFeedback", feedback_data)
        LAST_DELIVERY_FEEDBACK = feedback_date
        return feedback_data[0]

    def apply_coupon(self) -> Dict[str, Any]:
        """Apply a random coupon to the user's account."""
        # Get a random available coupon
        coupon = self.db.get_random_row("Coupons", columns=["coupon_id"])

        if not coupon:
            return None

        user_coupon_data = [
            {
                "user_id": self.data.user_id,
                "coupon_id": coupon["coupon_id"],
                "is_used": random.choice([True, False]),
            }
        ]

        self.db.insert("UserCoupons", user_coupon_data)
        return user_coupon_data[0]

    def schedule_order(self) -> Dict[str, Any]:
        """Schedule a future order from a random restaurant."""
        global LAST_SCHEDULED_ORDER
        if LAST_SCHEDULED_ORDER > datetime.now():
            return
        restaurant = self.db.get_random_row("Restaurants", columns=["restaurant_id"])

        # Set delivery time 1-24 hours after created_at
        hours_ahead = random.randint(1, 24)
        minutes = random.choice([0, 15, 30, 45])

        created_at = random_date_after_date(
            LAST_SCHEDULED_ORDER, timedelta_range=UTILITY_TIMEDELTA_RANGE
        )

        delivery_time = (created_at + timedelta(hours=hours_ahead)).replace(
            minute=minutes, second=0, microsecond=0
        )

        # Ensure delivery time is during reasonable hours (8 AM - 10 PM)
        while delivery_time.hour < 8 or delivery_time.hour > 22:
            delivery_time = delivery_time + timedelta(hours=1)

        scheduled_order_data = [
            {
                "user_id": self.data.user_id,
                "restaurant_id": restaurant["restaurant_id"],
                "delivery_time": delivery_time.strftime("%Y-%m-%d %H:%M:%S"),
                "created_at": created_at,
            }
        ]

        self.db.insert("ScheduledOrders", scheduled_order_data)
        LAST_SCHEDULED_ORDER = created_at
        return scheduled_order_data[0]

    def submit_app_feedback(self) -> Dict[str, Any]:
        """Submit feedback about the app."""
        global LAST_APP_REVIEW
        if LAST_APP_REVIEW > datetime.now():
            return
        feedback_data = [
            {
                "user_id": self.data.user_id,
                "feedback_text": random.choice(FEEDBACK_OPTIONS),
                "created_at": random_date_after_date(
                    LAST_APP_REVIEW, timedelta_range=UTILITY_TIMEDELTA_RANGE
                ),
            }
        ]
        self.db.insert("AppFeedback", feedback_data)
        LAST_APP_REVIEW = feedback_data[0]["created_at"]
        return feedback_data[0]

    def create_notification(self) -> Dict[str, Any]:
        """Create a notification for the user."""
        global LAST_NOTIFICATION_DATE
        if LAST_NOTIFICATION_DATE > datetime.now():
            return
        # Get a random order ID for reference
        order = self._get_order(None, ["order_id"])
        order_id = order["order_id"]
        # order_id = order["order_id"] if order else random.randint(1000, 9999)

        notification_data = [
            {
                "user_id": self.data.user_id,
                "message": random.choice(NOTIFICATIONS).format(order_id),
                "is_read": random.choice([True, False]),
                "created_at": random_date_after_date(
                    LAST_NOTIFICATION_DATE, timedelta_range=UTILITY_TIMEDELTA_RANGE
                ),
            }
        ]

        self.db.insert("Notifications", notification_data)
        LAST_NOTIFICATION_DATE = notification_data[0]["created_at"]
        return notification_data[0]

    def join_loyalty_program(self) -> Dict[str, Any]:
        """Enroll user in the loyalty program."""
        # Check if user is already enrolled
        existing_program = self.db.get_related_rows(
            "LoyaltyPrograms", "user_id", self.data.user_id, ["loyalty_id"]
        )

        if existing_program:
            return existing_program[0]

        initial_points = random.randint(0, 1000)
        tiers = ["Bronze", "Silver", "Gold", "Platinum"]
        tier_weights = [0.4, 0.3, 0.2, 0.1]  # Higher chance of lower tiers

        loyalty_data = [
            {
                "user_id": self.data.user_id,
                "points": initial_points,
                "tier": random.choices(tiers, weights=tier_weights)[0],
            }
        ]

        self.db.insert("LoyaltyPrograms", loyalty_data)
        return loyalty_data[0]

    def create_support_ticket(self) -> Dict[str, Any]:
        """Create a support ticket."""
        global LAST_SUPPORT_TICKET_DATE
        if LAST_SUPPORT_TICKET_DATE > datetime.now():
            return
        subjects = [
            "Order not delivered",
            "Wrong items in order",
            "Payment issue",
            "Account access problem",
            "Restaurant complaint",
            "App technical issue",
            "Refund status inquiry",
            "Delivery person complaint",
        ]

        descriptions = [
            "My order has been marked as delivered but I haven't received it.",
            "I received different items than what I ordered.",
            "My payment was processed twice for the same order.",
            "Unable to reset my password through the app.",
            "Restaurant cancelled my order without explanation.",
            "App keeps crashing when I try to place an order.",
            "Submitted refund request 3 days ago, no response yet.",
            "Delivery person was rude and unprofessional.",
        ]

        ticket_data = [
            {
                "user_id": self.data.user_id,
                "subject": random.choice(subjects),
                "description": random.choice(descriptions),
                "status": "Open",
                "created_at": random_date_after_date(
                    LAST_SUPPORT_TICKET_DATE, timedelta_range=UTILITY_TIMEDELTA_RANGE
                ),
            }
        ]

        self.db.insert("SupportTickets", ticket_data)
        LAST_SUPPORT_TICKET_DATE = ticket_data[0]["created_at"]
        return ticket_data[0]

    def perform_user_actions(self) -> None:
        """
        Perform a series of user actions based on trigger rates.

        Args:
            steps (int): Number of iterations to perform
        """
        # Define trigger rates for each action (in percentages)
        trigger_rates = {
            # "make_order": 100,  # Always create an order
            "request_refund": 10,
            "write_review": 10,
            "add_favorite": 1,
            "leave_delivery_feedback": 10,
            "apply_coupon": 10,
            "schedule_order": 5,
            "submit_app_feedback": 1,
            "create_notification": 5,
            "join_loyalty_program": 1,
            "create_support_ticket": 2,
        }

        # Map actions to their corresponding methods
        actions = {
            # "make_order": self.make_order, # BAS
            "request_refund": self.request_refund,  # GET LAST ORDER BAGLANTILI
            "write_review": self.write_review,  # GET LAST ORDER BAGLANTILI
            "add_favorite": self.add_favorite,  # GET LAST ORDER BAGLANTILI or pass ?
            "leave_delivery_feedback": self.leave_delivery_feedback,  # GET LAST ORDER BAGLANTILI
            "apply_coupon": self.apply_coupon,  # NO NEED
            "schedule_order": self.schedule_order,  # AYRI
            "submit_app_feedback": self.submit_app_feedback,  # AYRI
            "create_notification": self.create_notification,  # NO NEED AMA GERCEKCILIK ICIN YES NEED
            "join_loyalty_program": self.join_loyalty_program,  # NO NEED
            "create_support_ticket": self.create_support_ticket,  # AYRI
        }

        try:
            order_id = self.make_order()
        except Exception as e:
            return
        for action_name, trigger_rate in trigger_rates.items():
            if random.random() * 100 < trigger_rate:
                # try:
                action_method = actions[action_name]
                if action_name in [
                    "request_refund",
                    "write_review",
                    "leave_delivery_feedback",
                ]:
                    action_method(order_id=order_id)
                else:
                    action_method()
                # except Exception as e:
                # print(f"Error performing {action_name}: {str(e)}")


if __name__ == "__main__":
    config = Config("config.yaml")

    ## GET DB CONNECTION
    dataset = MysqlDataset(
        os.environ.get("MYSQL_HOST"),
        os.environ.get("MYSQL_PORT"),
        os.environ.get("MYSQL_USER"),
        os.environ.get("MYSQL_PASSWORD"),
    )

    query = """
        SELECT user_id, address 
        FROM Users 
        ORDER BY user_id
    """
    users = dataset.query_database(config.db_name, query)
    users = [User(dataset=dataset, db_name=config.db_name, **user) for user in users]

    step = 10000
    for _ in tqdm(range(step)):
        for user in random.choices(users, k=len(users) // 2):
            user.perform_user_actions()
