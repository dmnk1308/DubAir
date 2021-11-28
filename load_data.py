import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from helpers import in_one, drop_col, add_col
import ast

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

############# CATEGORICAL VARIABLES #################

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
    listings["bath_number"] = bath_number.astype(float)

    bath_kind = listings["bathrooms_text"].copy()

    shared = bath_kind.str.contains("shared", case = False)
    private = bath_kind.str.contains("private", case = False)
    normal = ~pd.concat([shared, private], axis = 1).any(axis = 1)

    bath_kind[shared] = "Shared"
    bath_kind[private] = "Private"
    bath_kind[normal] = "Normal"

    listings["bath_kind"] = bath_kind

    listings = listings.drop("bathrooms_text", axis = 1)

    # clean hotel, hostels
    prop = listings["property_type"]
    filter_prop = prop.str.contains("hotel", case = False)
    listings = listings[~filter_prop]
    price = price[~filter_prop]

    prop = listings["property_type"]
    filter_prop = prop.str.contains("hostel", case = False)
    listings = listings[~filter_prop]
    price = price[~filter_prop]

    # clean property types
    ## sum up all properties that occur less than 10 times in "others"
    values = listings["property_type"].value_counts()
    other_list = values.where(values<=10).dropna().index

    for i, j in enumerate(other_list):
        fil = listings["property_type"].str.contains(j, case = True, na = False)
        listings["property_type"][fil] = "Others"

####################### AMENITIES ###############################
    # load amenities
    amenities = listings["amenities"]

    # we hava a list as each cell of the amenities pd.Series. Unpack them
    amenities = amenities.apply(ast.literal_eval)
    mlb = MultiLabelBinarizer()
    am_array = mlb.fit_transform(amenities)
    am_df = pd.DataFrame(am_array, columns = mlb.classes_)

    # drop sum stuff that is too broad, too standard or to specific
    am_df = drop_col(am_df, "(Clothing storage)")
    am_df = drop_col(am_df, "(^Fast wifi.)")    

    am_df = drop_col(am_df, ["Bedroom comforts", "Bread maker","Carbon monoxide alarm",
    "Children’s dinnerware", "Drying rack for clothing", "Fireplace guards", "Fire extinguisher", 
    "Hot water kettle", "Hangers", "Iron", "Keypad", "Pocket wifi", "Mini fridge",
    "Mosquito net", "Outlet covers", "Pour-over coffee", "Portable fans",
    "Portable heater", "Portable air conditioning", "Radiant heating", "Record player", 
    "Rice maker", "Shower gel", "Ski-in/Ski-out", "Table corner guards", "Trash compactor",
    "Wine glasses", "Window guards", "Baking sheet", "Barbecue utensils", "Boat slip",
    "Cable TV","Changing table","Cleaning products","EV charger","Ethernet connection", 
    "Extra pillows and blankets", "First aid kit","Laundromat nearby", "Room-darkening shades",
    "Smart lock", "Smoke alarm", "Toaster", "Microwave", "Essentials", "Bathroom essentials", "Fire pit", 
    "Lock on bedroom door", "Hot water", "Beach essentials", "Board games", "Building staff", 
    "Cooking basics", "Dining table", "Dishes and silverware", "Host greets you", "Luggage dropoff allowed", 
    "Self check-in", "Pets allowed", "Suitable for events", "Ceiling fan"], regex = False)

    # sum up all luxury or extraordinary equipment
    am_df = in_one(am_df, ["Piano", "Ping pong table", "Kayak", "BBQ grill", "Bidet", "Bikes"], "Special_stuff", regex = False, sum = True, drop = True)

    # summarize in new columns which gives the total number
    am_df = in_one(am_df, "(HDTV)|(^\d\d..TV)|(^TV)", "TV_number", regex = True, sum = True, drop = True)
    am_df = in_one(am_df, "(game console)", "Oven_number", regex = True, sum = True, drop = True)
    am_df = in_one(am_df, "(^outdoor)", "Outdoor_stuff_number", regex = True, sum = True, drop = True)
    am_df = in_one(am_df, "(^Baby)|(^Crib$)|( crib$)|(^High chair$)", "Baby_friendly", regex = True, sum = True, drop = True)
    am_df = in_one(am_df, "(sound system)", "sound_system_number", regex = True, sum = True, drop = True)

    # summarize in new columns which gives the availability
    am_df = in_one(am_df, "(.oven)|(^oven)", "Oven_available", regex = True, sum = False, drop = True)
    am_df = in_one(am_df, "(.stove)|(^stove)", "Stoves_available", regex = True, sum = False, drop = True)
    am_df = in_one(am_df, "(refrigerator.)|(refrigerator)|(^Freezer$)", "Refridgerator_available", regex = True, sum = False, drop = True)
    am_df = in_one(am_df, "(body soap)", "Body_soap_available", regex = True, sum = False, drop = True)
    am_df = in_one(am_df, "(garden or backyard)|(^backyard)|(^garden)", "Garden_backyard_available", regex = True, sum = False, drop = True)
    am_df = in_one(am_df, "(^free.*parking)|(^free.*garage)|(^free.*carport)", "Free_parking_number", regex = True, sum = False, drop = True)
    am_df = in_one(am_df, "(^paid.*parking)|(^paid.*garage)|(^paid.*carport)", "Paid_parking_number", regex = True, sum = False, drop = True)
    am_df = in_one(am_df, "(^Children’s books and toys)", "Children_Entertainment", regex = True, sum = False, drop = True)
    am_df = in_one(am_df, "(^Dedicated workspace)", "Workspace", regex = True, sum = False, drop = True)
    am_df = in_one(am_df, "(conditioner)|(shampoo)", "Shampoo_Conditioner_available", regex = True, sum = False, drop = True)
    am_df = in_one(am_df, "(^Fast wifi.)", "Fast_wifi_available", regex = True, sum = False, drop = True)
    am_df = in_one(am_df, "(^Gym)|(. gym)", "Gym_available", regex = True, sum = False, drop = True)
    am_df = in_one(am_df, "(coffee machine)|(Nespresso)|(^Coffee maker)", "Coffee_machine_available", regex = True, sum = False, drop = True)
    am_df = in_one(am_df, "(^Dryer)|(Paid dryer)|(^Free dryer)", "Dryer_available", regex = True, sum = False, drop = True)
    am_df = in_one(am_df, "(^Washer)|(Paid washer)|(Free washer)", "Washer_available", regex = True, sum = False, drop = True)
    am_df = in_one(am_df, "(^Hot tub)|(.hot tub)", "Hot_tub_available", regex = True, sum = False, drop = True)
    am_df = in_one(am_df, "(^Pool)|(shared.*pool)|(private.*pool)", "Pool_available", regex = True, sum = False, drop = True)
    am_df = in_one(am_df, "(patio or balcony)", "Patio_balcony_available", regex = True, sum = False, drop = True)
    am_df = in_one(am_df, "(^Wifi)", "Wifi_available", regex = True, sum = False, drop = True)
    am_df = in_one(am_df, "(air conditioning)", "AC_available", regex = True, sum = False, drop = True)
    am_df = in_one(am_df, "(heating)", "heating_available", regex = True, sum = False, drop = True)
    am_df = in_one(am_df, "(^Kitchen$)|(^Full kitchen$)", "Kitchen_available", regex = True, sum = False, drop = True)
    am_df = in_one(am_df, "(^Lockbox$)|(^Safe$)", "Safe_available", regex = True, sum = False, drop = True)
    am_df = in_one(am_df, "(sauna)", "Sauna_available", regex = True, sum = False, drop = True)
    am_df = in_one(am_df, "(^Waterfront$)|(^Beachfront$)|(^Lake access$)", "Water_location", regex = True, sum = False, drop = True)


    am_df["Special_stuff"] = am_df["Special_stuff"] + am_df["Sauna_available"]

    am_df = am_df.drop("Sauna_available", axis = 1)

    # join amenities with listings
    listings = listings.join(am_df)
    # drop amenities columns
    listings = listings.drop("amenities", axis = 1)

    return price, listings, reviews