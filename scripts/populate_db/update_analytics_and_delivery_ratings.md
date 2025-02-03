## Query to Update Restaurant Analytics based on Orders
```
UPDATE RestaurantAnalytics ra 
LEFT JOIN (
    SELECT 
        restaurant_id,
        COUNT(*) as order_count,
        COALESCE(SUM(total_price), 0) as total_rev
    FROM Orders
    GROUP BY restaurant_id
) o ON ra.restaurant_id = o.restaurant_id
LEFT JOIN (
    SELECT 
        restaurant_id,
        COALESCE(AVG(rating), 0) as avg_rate
    FROM Reviews
    GROUP BY restaurant_id
) r ON ra.restaurant_id = r.restaurant_id
SET 
    ra.total_orders = COALESCE(o.order_count, 0),
    ra.total_revenue = COALESCE(o.total_rev, 0),
    ra.avg_rating = COALESCE(r.avg_rate, 0)
WHERE ra.analytics_id > 0;
```

## Query to Insert Delivery Ratings, TRUNCATE TABLE FIRST
```
INSERT INTO DeliveryRatings (delivery_person_id, rating)
SELECT 
    d.delivery_person_id,
    COALESCE(AVG(df.rating), 0) as avg_rating
FROM DeliveryPersons dp
LEFT JOIN Deliveries d ON dp.delivery_person_id = d.delivery_person_id
LEFT JOIN DeliveryFeedback df ON d.delivery_id = df.delivery_id
GROUP BY dp.delivery_person_id;
```
