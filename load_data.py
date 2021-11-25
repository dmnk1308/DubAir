import pandas as pd
def load_data():
    url_listing = "http://data.insideairbnb.com/ireland/leinster/dublin/2021-11-07/data/listings.csv.gz"
    url_reviews = "http://data.insideairbnb.com/ireland/leinster/dublin/2021-11-07/data/reviews.csv.gz"
    listings = pd.read_csv(url_listing)
    reviews = pd.read_csv(url_reviews)
    variables_listing= ["name", "description", "neighborhood_overview", "host_name", "host_since", "host_location", 
    "host_about", "host_is_superhost", "host_listings_count", "host_has_profile_picture","host_identity_verified",
    "neighbourhood_cleansed",
    "latitude",
    "longitude",
    "property type",
    "room type",
    "accommodates",
    "bathrooms_text",
    "bedrooms",
    "beds",
    "amenities",
    "minimum_nights",
    "maximum_nights",
    "has_availability",
    "availability_30",
    "availability_60",
    "availability_90",
    "availability_365",
    "number_of_reviews",
    "number_of_reviews_ltm", 
    "number_of_reviews_l30d", 
    "first_review",
    "last_review",
    "review_scores_rating",	 
    "review_scores_accuracy",	
    "review_scores_cleanliness",
    "review_scores_checkin	",
    "review_scores_communication",
    "review_scores_location",	
    "review_scores_value",
    "instant_bookable",
    "calculated_host_listings_count",
    "reviews_per_month"] 
    price = listings["price"]
    price = price.str.replace("$","")
    price = price.str.replace(",","")
    price = price.astype(float)

    listings = listings.filter(variables_listing)
    reviews = reviews.filter(["comments","date"])
    return price, listings, reviews