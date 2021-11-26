import pandas as pd

def load_data():
    url_listing = "http://data.insideairbnb.com/ireland/leinster/dublin/2021-11-07/data/listings.csv.gz"
    url_reviews = "http://data.insideairbnb.com/ireland/leinster/dublin/2021-11-07/data/reviews.csv.gz"
    listings = pd.read_csv(url_listing)
    reviews = pd.read_csv(url_reviews)
    variables_listing= ["name", "last_scraped", "description", "neighborhood_overview", "host_name", "host_since", "host_location", 
    "host_about", "host_is_superhost", "host_listings_count", "host_has_profile_picture","host_identity_verified",
    "neighbourhood_cleansed",
    "latitude",
    "longitude",
    "property_type",
    "room_type",
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
    # make price numeric
    price = listings["price"]
    price = price.str.replace("$","")
    price = price.str.replace(",","")
    price = price.astype(float)

    # get rid of Hotels
    hotel_filter = listings["room_type"] == "Hotel room"
    listings = listings[~hotel_filter]
    price = price[~hotel_filter]
    

    listings = listings.filter(variables_listing)
    reviews = reviews.filter(["comments","date"])

    return price, listings, reviews



def load_data_cleansed():
    price, listings, reviews = load_data()


    # clean host_location
    country_abr = pd.read_csv("https://gist.githubusercontent.com/radcliff/f09c0f88344a7fcef373/raw/2753c482ad091c54b1822288ad2e4811c021d8ec/wikipedia-iso-country-codes.csv")
    country_list = list(country_abr.iloc[:,0])
    abr_list = list(country_abr.iloc[:,1])

    listings["host_location_country"] = listings["host_location"].copy()

    for i in list(country_list):
        fil = listings["host_location"].str.contains(i, case = False, na = False)
        listings["host_location_country"][fil] = str(i)

    for i,j in enumerate(list(abr_list)):
        fil = listings["host_location"].str.contains(str(j), case = True, na = False)
        listings["host_location_country"][fil] = str(country_list[i])

    listings["host_location_country"].value_counts()

    other_filter = listings["host_location_country"].value_counts() <= 5
    other_list = list(listings["host_location_country"].value_counts().index[other_filter])

    for i, j in enumerate(other_list):
        fil = listings["host_location_country"].str.contains(j, case = True, na = False)
        listings["host_location_country"][fil] = "Others"
    listings["host_location_country"][listings["host_location_country"] == "53.357852, -6.259787"] = "Ireland"
    
    listings = listings.drop("host_location", axis = 1)

    # clean bathroom text
    na_filter = listings["bathrooms_text"].isna()
    price = price[~na_filter]
    listings = listings[~na_filter]

    bath_number = listings["bathrooms_text"].copy()
    bath_number = bath_number.str.replace("half", "0.5", case = False)
    bath_number = bath_number.str.extract('(\d+.\d|\d+)')
    listings["bath_number"] = bath_number

    bath_kind = listings["bathrooms_text"].copy()

    shared = bath_kind.str.contains("shared", case = False)
    private = bath_kind.str.contains("private", case = False)
    normal = ~pd.concat([shared, private], axis = 1).any(axis = 1)

    bath_kind[shared] = "Shared"
    bath_kind[private] = "Private"
    bath_kind[normal] = "Normal"

    listings["bath_kind"] = bath_kind

    listings = listings.drop("bathrooms_text", axis = 1)

    
    return price, listings, reviews