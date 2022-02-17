import os
os.chdir("/Users/dmnk/OneDrive - stud.uni-goettingen.de/Dokumente/3. Semester/SeminarDL/DubAir")
import numpy as np
from helpers import *
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer, OneHotEncoder
from sklearn.feature_selection import VarianceThreshold
import ast
import requests
from bs4 import BeautifulSoup as bs
import statsmodels.api as sm
from scipy.stats import halfnorm
import warnings
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from scipy import stats
import matplotlib.pyplot as plt
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform



class Wrangler:
    
    
    def __init__(self):
        self.relevant_variables = ["id", "name", "last_scraped", "description", "neighborhood_overview", "host_id", "host_url", "host_name", "host_since", "host_location",
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
                                "reviews_per_month",
                                "host_has_profile_pic",
                                'minimum_minimum_nights', 
                                'maximum_minimum_nights', 
                                'minimum_maximum_nights', 
                                'maximum_maximum_nights', 
                                'minimum_nights_avg_ntm', 
                                'maximum_nights_avg_ntm',
                                'calculated_host_listings_count_entire_homes', 
                                'calculated_host_listings_count_private_rooms', 
                                'calculated_host_listings_count_shared_rooms', "price"] 
        
    def preprocess(self):
        self.data = self.data.reset_index(drop = True)

        # get rid of Hotels
        hotel_filter = self.data["room_type"] == "Hotel room"
        self.data = self.data[~hotel_filter]
        # clean hotel, hostels again
        prop = self.data["property_type"]
        filter_prop = prop.str.contains("hotel", case = False)
        self.data = self.data[~filter_prop]
        prop = self.data["property_type"]
        filter_prop = prop.str.contains("hostel", case = False)
        self.data = self.data[~filter_prop]
         # remove everything unimportant
        self.data = self.data.filter(self.relevant_variables)
        
        
        #### Preprocess ####
        
        # clean host_profile_pic
        self.data["host_has_profile_pic"] = np.where(self.data["host_has_profile_pic"] == "t", 1, 0)
        self.data["host_is_superhost"] = np.where(self.data["host_is_superhost"] == "t", 1, 0)
        self.data["host_identity_verified"] = np.where(self.data["host_identity_verified"] == "t", 1, 0)
        self.data["has_availability"] = np.where(self.data["has_availability"] == "t", 1, 0)
        self.data["instant_bookable"] = np.where(self.data["instant_bookable"] == "t", 1, 0)


        # clean bathroom text
        na_filter = self.data["bathrooms_text"].isna()
        self.data = self.data[~na_filter]
        bath_number = self.data["bathrooms_text"].copy()
        bath_number = bath_number.str.replace("half", "0.5", case = False)
        bath_number = bath_number.str.extract('(\d+.\d|\d+)')
        self.data["bath_number"] = bath_number.astype(float)
        bath_kind = self.data["bathrooms_text"].copy()
        shared = bath_kind.str.contains("shared", case = False)
        private = bath_kind.str.contains("private", case = False)
        normal = ~pd.concat([shared, private], axis = 1).any(axis = 1)
        bath_kind[shared] = "Shared"
        bath_kind[private] = "Private"
        bath_kind[normal] = "Normal"
        self.data["bath_kind"] = bath_kind
        self.data = self.data.drop("bathrooms_text", axis = 1)

        # clean property types
        ## sum up all properties that occur less than 10 times in "others"
        values = self.data["property_type"].value_counts()
        other_list = values.where(values<=10).dropna().index
        for i, j in enumerate(other_list):
            fil = self.data["property_type"].str.contains(j, case = True, na = False)
            self.data.loc[fil,"property_type"] = "Others"

        return self.data
    
    def process_amenities(self, fit = True):
        self.data = self.data.reset_index(drop = True)
        # AMENITIES
        # load amenities
        amenities = self.data["amenities"]
        amenities = amenities.apply(ast.literal_eval)

        # we hava a list as each cell of the amenities pd.Series. Unpack them
        if fit:
            mlb_amenities = MultiLabelBinarizer()
            mlb_amenities.fit(amenities)
            self.mlb_amenities = mlb_amenities
        
        am_array = self.mlb_amenities.transform(amenities)
        am_df = pd.DataFrame(am_array, columns = self.mlb_amenities.classes_)
        # drop some stuff that is too broad, too standard or to specific
        am_df = drop_col(am_df, "(Clothing storage)")
        am_df = drop_col(am_df, "(^Fast wifi.)")    

        # am_df = drop_col(am_df, ["Bedroom comforts", "Bread maker","Carbon monoxide alarm",
        # "Children’s dinnerware", "Drying rack for clothing", "Fireplace guards", "Fire extinguisher", 
        # "Hot water kettle", "Hangers", "Iron", "Keypad", "Pocket wifi", "Mini fridge",
        # "Mosquito net", "Outlet covers", "Pour-over coffee", "Portable fans",
        # "Portable heater", "Portable air conditioning", "Radiant heating", "Record player", 
        # "Rice maker", "Shower gel", "Ski-in/Ski-out", "Table corner guards", "Trash compactor",
        # "Wine glasses", "Window guards", "Baking sheet", "Barbecue utensils", "Boat slip",
        # "Cable TV","Changing table","Cleaning products","EV charger","Ethernet connection", 
        # "Extra pillows and blankets", "First aid kit","Laundromat nearby", "Room-darkening shades",
        # "Smart lock", "Smoke alarm", "Toaster", "Microwave", "Essentials", "Bathroom essentials", "Fire pit", 
        # "Lock on bedroom door", "Hot water", "Beach essentials", "Board games", "Building staff", 
        # "Cooking basics", "Dining table", "Dishes and silverware", "Host greets you", "Luggage dropoff allowed", 
        # "Self check-in", "Pets allowed", "Suitable for events", "Ceiling fan"], regex = False)

        # summarize in new columns which gives the availability
        am_df = in_one(am_df, "(.oven)|(^oven)", "Oven_available", regex = True, sum = False, drop = True)
        am_df = in_one(am_df, "(.stove)|(^stove)", "Stoves_available", regex = True, sum = False, drop = True)
        am_df = in_one(am_df, "(refrigerator.)|(refrigerator)|(^Freezer$)", "Refridgerator_available", regex = True, sum = False, drop = True)
        am_df = in_one(am_df, "(body soap)", "Body_soap_available", regex = True, sum = False, drop = True)
        am_df = in_one(am_df, "(garden or backyard)|(^backyard)|(^garden)", "Garden_backyard_available", regex = True, sum = False, drop = True)
        am_df = in_one(am_df, "(^free.*parking)|(^free.*garage)|(^free.*carport)", "Free_parking", regex = True, sum = False, drop = True)
        am_df = in_one(am_df, "(^paid.*parking)|(^paid.*garage)|(^paid.*carport)", "Paid_parking", regex = True, sum = False, drop = True)
        am_df = in_one(am_df, "(^Children’s books and toys)", "Children_Entertainment", regex = True, sum = False, drop = True)
        am_df = in_one(am_df, "(^Dedicated workspace)", "Workspace", regex = True, sum = False, drop = True)
        am_df = in_one(am_df, "(conditioner)|(shampoo)", "Shampoo_Conditioner_available", regex = True, sum = False, drop = True)
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
        am_df = in_one(am_df, "(sound system)", "sound_system_available", regex = True, sum = False, drop = True)
        am_df = in_one(am_df, "(HDTV)|(^\d\d..TV)|(^TV)", "TV_available", regex = True, sum = False, drop = True)
        am_df = in_one(am_df, "(^outdoor)", "Outdoor_stuff", regex = True, sum = False, drop = True)
        am_df = in_one(am_df, "(game console)", "Game_consoles", regex = True, sum = False, drop = True)
        am_df = in_one(am_df, "(^Baby)|(^Crib$)|( crib$)|(^High chair$)", "Baby_friendly", regex = True, sum = False, drop = True)

        # sum up all luxury or extraordinary equipment
        am_df = in_one(am_df, ["Piano", "Ping pong table", "Kayak", "BBQ grill", "Bidet", "Bikes", "Sauna_available"], "Special_stuff", regex = False, sum = False, drop = True)


        if fit:
            sel = VarianceThreshold(threshold=(.9 * (1 - .9)))
            sel.feature_names_in_ = am_df.columns
            self.variance_threshold_am = sel.fit(am_df)
        
        am_col = self.variance_threshold_am.get_feature_names_out()
        print(str(len(am_df.columns) - len(am_col)) + " amenities have been removed due to close zero-variance.")
        am_df = am_df.filter(am_col)
        # join amenities with listings

        self.data = pd.concat([self.data, am_df], axis = 1)
        # drop amenities columns
        self.data = self.data.drop("amenities", axis = 1)
        
        return self.data
        
    def add_stuff(self, munich = False):
        
        # ADD TEXT STUFF
        # lengths of text columns
        self.data["name_length"] = self.data["name"].astype(str).str.replace(" ","").str.len()
        self.data["description_length"] = self.data["description"].astype(str).str.replace(" ","").str.len()
        self.data["neighborhood_overview_length"] = self.data["neighborhood_overview"].astype(str).str.replace(" ","").str.len()
        self.data["host_about_length"] = self.data["host_about"].astype(str).str.replace(" ","").str.len()
       
        if munich == True:
            # read in pre-created frames
            listings_reviews = pd.read_csv("munich/listings_reviews_munich.csv")
            host_sent = pd.read_csv("munich/host_sent_munich.csv")
            host_name = pd.read_csv("munich/host_name_munich.csv")
            host_sent = host_sent.drop(host_sent.columns[0], axis=1)
            host_name = host_name.drop(host_name.columns[0], axis=1)        
            listings_reviews = listings_reviews.drop(listings_reviews.columns[0], axis=1)

            # add to listings
            self.data = pd.merge(self.data, listings_reviews, on="id", how="left")
            host_sent = pd.concat([host_sent, host_name], axis=1)
            self.data = pd.merge(self.data, host_sent, on="id", how="left")

            # ADD OSM STUFF
            listings_osm = pd.read_csv("munich/StreetData_munich.csv")
            listings_osm = listings_osm.drop(listings_osm.columns[0], axis=1)
            self.data = pd.merge(self.data, listings_osm, on="id", how="left")
    
        if munich == False:
            # read in pre-created frames
            listings_reviews = pd.read_csv("text_data/listings_reviews.csv")
            host_sent = pd.read_csv("text_data/host_sent.csv")
            host_name = pd.read_csv("text_data/host_name.csv")
            host_sent = host_sent.drop(host_sent.columns[0], axis=1)
            host_name = host_name.drop(host_name.columns[0], axis=1)        
            listings_reviews = listings_reviews.drop(listings_reviews.columns[0], axis=1)

            # add to listings
            self.data = pd.merge(self.data, listings_reviews, on="id", how="left")
            host_sent = pd.concat([host_sent, host_name], axis=1)
            self.data = pd.merge(self.data, host_sent, on="id", how="left")

            # ADD OSM STUFF
            listings_osm = pd.read_csv("StreetData.csv")
            listings_osm = listings_osm.drop(listings_osm.columns[0], axis=1)
            self.data = pd.merge(self.data, listings_osm, on="id", how="left")
            
        # ADD IMAGE STUFF
        img_df = pd.read_csv("data/img_info.csv")
        self.data = self.data.merge(img_df, how = "left", on = "id")
        self.data.drop("index", axis =1, inplace = True)
        print("Text, OpenStreet and image data loaded.")
        return self.data

    def fit_first(self):
        # IMPUTATION STUFF
        # FIT MODEL FOR BEDS
        # accomodates and beds are quite linear
        # So let us estimate linear models and predict, for beds
        Y = self.data["beds"]
        x = self.data["accommodates"]
        X = pd.DataFrame([x]).transpose()
        X = sm.add_constant(X)  # adding a constant
        self.model_OLS_beds = sm.OLS(Y, X, missing='drop').fit()

        # FIT MODEL FOR BEDROOMS
        # beds and bedrooms are very linear as well
        # do the same here
        Y = self.data["bedrooms"]
        x = self.data["beds"]
        X = pd.DataFrame([x]).transpose()
        X = sm.add_constant(X)  # adding a constant
        self.model_OLS_bedrooms = sm.OLS(Y, X, missing='drop').fit()

        # SAVE SD FOR EACH REVIEW VARIABLE
        # All those review score variables
        self.review_var = ['review_scores_rating', 'review_scores_accuracy', 'review_scores_cleanliness',
                    'review_scores_communication', 'review_scores_location', 'review_scores_value']
        # they look quite halfnormal
        sds = []
        for i in range(len(self.review_var)):
            sd = np.nanstd(self.data[self.review_var[i]],  ddof=0)  # ML-estimator
            sds.append(sd - sd / (4 * len(self.data[self.review_var[i]])))  # MLE bias corrected          
        self.sd_reviews = sds    
        
        # HOST LOCATION
        # clean host_location
        country_abr = pd.read_csv("https://gist.githubusercontent.com/radcliff/f09c0f88344a7fcef373/raw/2753c482ad091c54b1822288ad2e4811c021d8ec/wikipedia-iso-country-codes.csv")
        country_list = list(country_abr.iloc[:,0])
        abr_list = list(country_abr.iloc[:,1])

        self.data["host_location_country"] = self.data["host_location"].copy()

        for i in list(country_list):
            fil = self.data["host_location"].str.contains(i, case = False, na = False)
            self.data.loc[fil,"host_location_country"] = str(i)

        for i,j in enumerate(list(abr_list)):
            fil = self.data["host_location"].str.contains(str(j), case = True, na = False)
            self.data.loc[fil,"host_location_country"] = str(country_list[i])

        other_filter = self.data["host_location_country"].value_counts() <= 5
        self.country_list = list(self.data["host_location_country"].value_counts().index[other_filter])

        for i, j in enumerate(self.country_list):
            fil = self.data["host_location_country"].str.contains(j, case = True, na = False)
            self.data.loc[fil,"host_location_country"] = "Others"
        self.data.loc[self.data["host_location_country"] == "53.357852, -6.259787", "host_location_country"] = "Ireland"
        
        # PIs FOR BINARY VARIABLES
        ### Binary Stuff
        # self.rest_var = ['Bathtub', 'Bed linens', 'Breakfast', 'Cleaning before checkout', 'Dishwasher',
        #             'Elevator', 'Hair dryer', 'Indoor fireplace', 'Long term stays allowed',
        #             'Private entrance', 'Security cameras on property', 'Single level home',
        #             'Special_stuff', 'TV_available', 'Outdoor_stuff', 'Baby_friendly',
        #             'sound_system_available', 'Oven_available', 'Stoves_available',
        #             'Refridgerator_available', 'Body_soap_available',
        #             'Garden_backyard_available', 'Free_parking',
        #             'Paid_parking', 'Children_Entertainment', 'Workspace',
        #             'Shampoo_Conditioner_available', 'Gym_available',
        #             'Coffee_machine_available', 'Dryer_available', 'Washer_available',
        #             'Hot_tub_available', 'Pool_available', 'Patio_balcony_available',
        #             'Wifi_available', 'AC_available', 'heating_available',
        #             'Kitchen_available', 'Safe_available', 'Water_location', "Game_consoles"]
        
        self.rest_var = [col for col in self.data if np.isin(self.data[col].unique(), [0, 1]).all()]

        pis = []
        for i in range(len(self.rest_var)):
            pis.append(np.nanmean(self.data[self.rest_var[i]]))
        self.pis = pis
        
        # MEANS FOR TEXT STUFF
        self.text_var = ["prop_of_eng_reviews", "mean_compound", "mean_negativity", "mean_neutrality","mean_positivity","mean_review_length",
                         "prop_of_neg_comp", "most_neg_compound","most_pos_compound"]
        means_text = []
        for i in self.text_var:
            means_text.append(self.data[i].mean())
        self.means_text = means_text
            
        # MEANS FOR IMAGE STUFF
        img_df = pd.read_csv("data/img_info.csv")
        means = img_df.mean(axis = 0)
        self.mean_brightness = means[2]
        self.mean_contrast = means[3]  
        
        
        ### APPLY HERE FOR FOLLOWING FITS
        # OLS BEDS
        ind = self.data[self.data["beds"].isna()]["beds"].index
        x0 = self.data["accommodates"][ind]
        x0 = sm.add_constant(x0)
        predictions = self.model_OLS_beds.predict(x0)
        prediction = np.where(predictions <= 0.5, 1, predictions)   
        self.data.loc[ind, "beds"] = round(predictions)
        
        # OLS BEDROOMS
        ind = self.data[self.data["bedrooms"].isna()]["bedrooms"].index
        x0 = self.data["beds"][ind]
        x0 = sm.add_constant(x0)
        predictions = self.model_OLS_bedrooms.predict(x0)
        prediction = np.where(predictions <= 0.5, 1, predictions)   
        self.data.loc[ind, "bedrooms"] = round(predictions)          
        
        # ONE HOT
        listings_fit = self.data.copy()
        listings_fit["bath_number"] = np.round(listings_fit["bath_number"], 0).astype(int)
        listings_fit["bath_number"] = np.where(listings_fit["bath_number"] > 3, 4, listings_fit["bath_number"]).astype(str)
        listings_fit["bedrooms"] = np.round(listings_fit["bedrooms"], 0).astype(int)
        listings_fit["bedrooms"] = np.where(listings_fit["bedrooms"] > 3, 4, listings_fit["bedrooms"]).astype(str)
        self.one_hot_columns = ["bath_number", "bedrooms", "host_location_country", "neighbourhood_cleansed", "property_type", "room_type", "bath_kind"]        
        self.one_hot = OneHotEncoder(handle_unknown="ignore")
        self.one_hot.fit(listings_fit[self.one_hot_columns])

        return self
    
    def transform_first(self, fit = True, munich = False):   
        # CLEAN HOST LOCATION
        country_abr = pd.read_csv("https://gist.githubusercontent.com/radcliff/f09c0f88344a7fcef373/raw/2753c482ad091c54b1822288ad2e4811c021d8ec/wikipedia-iso-country-codes.csv")
        country_list = list(country_abr.iloc[:,0])
        abr_list = list(country_abr.iloc[:,1])
        self.data["host_location_country"] = self.data["host_location"].copy()
        for i in list(country_list):
            fil = self.data["host_location"].str.contains(i, case = False, na = False)
            self.data.loc[fil,"host_location_country"] = str(i)
        for i,j in enumerate(list(abr_list)):
            fil = self.data["host_location"].str.contains(str(j), case = True, na = False)
            self.data.loc[fil,"host_location_country"] = str(country_list[i])
        self.data = self.data.reset_index(drop = True)
        for i, j in enumerate(self.country_list):
            fil = self.data["host_location_country"].str.contains(j, case = True, na = False)
            self.data.loc[fil,"host_location_country"] = "Others"
        self.data.loc[self.data["host_location_country"] == "53.357852, -6.259787", "host_location_country"] = "Ireland"
        
        # IMPUTATION 
        # name and description, take room_type instead
        ind = self.data[self.data["name"].isna()]["name"].index
        self.data.loc[ind, "name"] = self.data.loc[ind, "room_type"]

        ind = self.data[self.data["description"].isna()]["description"].index
        self.data.loc[ind, "description"] = self.data.loc[ind, "room_type"]
        
        # neighbourhood-overview (=description) just neihgbourhood cleansed
        ind = self.data[self.data["neighborhood_overview"].isna()]["neighborhood_overview"].index
        self.data.loc[ind, "neighborhood_overview"] = self.data.loc[ind, "neighbourhood_cleansed"]
        # host_about
        ind = self.data[self.data["host_about"].isna()]["host_about"].index
        self.data.loc[ind, "host_about"] = " "

        # first and last review, you might want to think about this again
        ind = self.data[self.data["first_review"].isna()]["first_review"].index
        self.data.loc[ind, "first_review"] = self.data.loc[ind, "last_scraped"]

        ind = self.data[self.data["last_review"].isna()]["last_review"].index
        self.data.loc[ind, "last_review"] = self.data.loc[ind, "last_scraped"]

        # Reviews per Month are probably zero
        ind = self.data[self.data["reviews_per_month"].isna()]["reviews_per_month"].index
        self.data.loc[ind, "reviews_per_month"] = self.data.loc[ind, "number_of_reviews"]

        if munich:
            # If the host_location is not given, they are probably in Germany
            ind = self.data[self.data["host_location_country"].isna()]["host_location_country"].index
            self.data.loc[ind, "host_location_country"] = "Germany"
            
        else:
            # If the host_location is not given, they are probably in Ireland
            ind = self.data[self.data["host_location_country"].isna()]["host_location_country"].index
            self.data.loc[ind, "host_location_country"] = "Ireland"

        ## Some webscraping for host-variables -> shall be the same profiles
        ind_s = self.data[self.data["host_name"].isna()]["host_name"].index
        rel_URL = self.data.loc[ind_s, "host_url"]
        ids = self.data.loc[ind_s, "host_id"]

        name = []
        id_ver = []
        for i in range(len(ind_s)):
            self.data.loc[ind_s, "host_listings_count"] = len(self.data[self.data.host_id == ids.values[i]])
            session = requests.Session()
            html_code = session.get(rel_URL.values[i]).content
            soup = bs(html_code, "html.parser")
            name_html = soup.select("._a0kct9 ._14i3z6h")
            # the if statement is for profiles that cannot be called for any reason
            if len(name_html) == 0:
                name.append("Anonymous")
            else:
                name.append(name_html[0].text[8:])

        self.data.loc[ind_s, "host_name"] = name
        self.data.loc[ind_s, "host_since"] = self.data.loc[ind_s, "first_review"]
    
        # OLS BEDS
        ind = self.data[self.data["beds"].isna()]["beds"].index
        x0 = self.data["accommodates"][ind]
        x0 = sm.add_constant(x0)
        predictions = self.model_OLS_beds.predict(x0)
        prediction = np.where(predictions <= 0.5, 1, predictions)   
        self.data.loc[ind, "beds"] = round(predictions) 
        
        # OLS BEDROOMS
        ind = self.data[self.data["bedrooms"].isna()]["bedrooms"].index
        x0 = self.data["beds"][ind]
        x0 = sm.add_constant(x0)
        predictions = self.model_OLS_bedrooms.predict(x0)
        prediction = np.where(predictions <= 0.5, 1, predictions)   
        self.data.loc[ind, "bedrooms"] = round(predictions)
        
        # REVIEWS HALFNORMAL
        for i in range(len(self.review_var)):
            ind = self.data[self.data[self.review_var[i]].isna()][self.review_var[i]].index
            np.random.seed(123)
            fill_ind = (halfnorm.rvs(loc=0, scale=self.sd_reviews[i], size=len(ind)) * -1) + 5
            self.data.loc[ind, self.review_var[i]] = fill_ind
            
        # BINARY IMPUTATION 
        for i in range(len(self.rest_var)):
            ind = self.data[self.data[self.rest_var[i]].isna()][self.rest_var[i]].index
            self.data.loc[ind, self.rest_var[i]] = np.random.binomial(n=1, p=self.pis[i], size=len(ind))
        
        # IMPUTATION TEXT STUFF
        for j,i in enumerate(self.text_var):
            self.data[i].fillna(self.means_text[j], inplace=True)
                       
        # IMPUTATION IMAGE STUFF
        img_df = pd.read_csv("data/img_info.csv")       
        self.room_cols = ["no_img_bathroom","no_img_bedroom","no_img_dining","no_img_hallway","no_img_kitchen","no_img_living","no_img_others"] #"no_img_balcony",
        self.data["count"] = self.data["count"].fillna(0)
        self.data["brightness"] = self.data["brightness"].fillna(self.mean_brightness)
        self.data["contrast"] = self.data["contrast"].fillna(self.mean_contrast)
        if munich:
            for i in self.room_cols:
                self.data[i] = img_df[i].mean()
        else:
            self.data[self.room_cols] = self.data[self.room_cols].fillna(0)
        
        # ONE HOT
        self.data["bath_number"] = np.round(self.data["bath_number"], 0).astype(int)
        self.data["bath_number"] = np.where(self.data["bath_number"] > 3, 4, self.data["bath_number"]).astype(str)
        self.data["bedrooms"] = np.round(self.data["bedrooms"], 0).astype(int)
        self.data["bedrooms"] = np.where(self.data["bedrooms"] > 3, 4, self.data["bedrooms"]).astype(str)
        one_hots = self.one_hot.transform(self.data[self.one_hot_columns]).toarray().astype(int)
        one_hots = pd.DataFrame(one_hots)
        one_hots.columns = self.one_hot.get_feature_names_out(input_features = self.one_hot_columns)
        self.data = pd.concat([self.data, one_hots], axis=1)
        self.data.drop(self.one_hot_columns, axis = 1, inplace = True)

        # TIME VARIABLES
        date_col = ["last_scraped", "host_since", "first_review", "last_review"]
        pd.to_datetime(self.data["last_scraped"], yearfirst=True)
        date_df = self.data.filter(date_col).apply(pd.to_datetime)
        self.data["host_since"] = date_df["last_scraped"] - date_df["host_since"]
        self.data["first_review"] = date_df["last_scraped"] - date_df["first_review"]
        self.data["last_review"] = date_df["last_scraped"] - date_df["last_review"]
        self.data = self.data.drop("last_scraped", axis=1)
        # We have a timedelta object in each cell now. We should convert it into an integer using its attribute .days
        date_col = date_col[1:]
        for i in date_col:
            self.data[i] = pd.Series([j.days for j in list(self.data[i])])
        
        # VARIANCE THRESHOLD
       
        
        if fit:
            bin_col = [col for col in self.data if np.isin(self.data[col].unique(), [0, 1]).all()]
            num_col = [col for col in self.data if ~np.isin(self.data[col].unique(), [0, 1]).all()]
            binary_df = self.data.filter(bin_col)
            sel = VarianceThreshold(threshold=(.9 * (1 - .9)))
            sel.feature_names_in_ = binary_df.columns
            self.variance_threshold = sel.fit(binary_df)
            binary_col = self.variance_threshold.get_feature_names_out()
            all_col = binary_col.tolist() + num_col
            all_col = np.unique(np.array(all_col)).tolist()        
            self.all_col_var = all_col
        print(str(len(self.data.columns) - len(self.all_col_var)) + " binary variables have been removed due to close zero-variance.")
        
        self.data = self.data.filter(self.all_col_var)  

        # DROP
        self.data = self.data.drop(["host_location","host_id", "host_url", "name", "description", "neighborhood_overview", "host_name", "host_about"], axis = 1)

        # CHECK FOR NaNs
        if len(self.data.isna().sum()[self.data.isna().sum().values > 0]) == 0:
            print("Imputation done. No NaN's are left in the data.")
        else:
            print("Imputation failed. There are NaN's left; here is where:")
            print(self.data.isna().sum()[self.data.isna().sum().values > 0])

        return self.data
    
    def fit_second(self):
        # PCAs FIT
        self.city_life = ["nightclubs", "sex_amenities", "bicycle_rentals", "casinos", "university",     
                          "theatres_artscentre", "library", "taxi", "fast_foods", "restaurants", "bars",
                          "cafes", "malls", "cinemas", "supermarkets", "bus_train_tram_station", "social_amenities"]
        scaler = StandardScaler()
        self.scaler_pca_city_life = scaler.fit(self.data[self.city_life])
        city_life_df = self.scaler_pca_city_life.transform(self.data[self.city_life])
        self.pca_city = PCA(n_components = 5).fit(city_life_df)
        
        # PCA for touristic and travel
        self.travel_touristic = ["neighbourhood_cleansed_Dublin City", "in_city", "nearest_sight", "mean_dist_sight", 
                                 "2nd_nearest_sight", "3rd_nearest_sight", "nearest_travel_poss", "mean_dist_travel"]

        scaler = StandardScaler()
        self.scaler_pca_travel = scaler.fit(self.data[self.travel_touristic])
        travel_touristic_df = self.scaler_pca_travel.transform(self.data[self.travel_touristic])
        self.pca_travel = PCA(n_components = 1).fit(travel_touristic_df)

        # PCA for kitchen + equipment
        self.kitchen = ["Microwave", "Dishes and silverware", "Refridgerator_available", "Dishwasher",
                        "Stoves_available", "Cooking basics", "Oven_available", "Kitchen_available"]

        scaler = StandardScaler()
        self.scaler_pca_kitchen = scaler.fit(self.data[self.kitchen])
        kitchen_df = self.scaler_pca_kitchen.transform(self.data[self.kitchen])
        self.pca_kitchen = PCA(n_components = 4).fit(kitchen_df)

        # PCA for accommodation size
        self.acco = ["bedrooms_1", "bedrooms_2", "accommodates", "beds", "room_type_Entire home/apt", "room_type_Private room",
                     "bath_number_1", "bath_number_2", "bath_kind_Shared", "bath_kind_Private", "bath_kind_Normal",
                     "property_type_Entire residential home", "property_type_Entire rental unit", "property_type_Others"]
        scaler = StandardScaler()
        self.scaler_pca_acco = scaler.fit(self.data[self.acco])
        accommodation_size_df = self.scaler_pca_acco.transform(self.data[self.acco])
        self.pca_acco_size = PCA(n_components = 6).fit(accommodation_size_df)
        
        # PCA for host listings counts
        self.host_listings = ["calculated_host_listings_count", "host_listings_count", 
                              "calculated_host_listings_count_private_rooms", "calculated_host_listings_count_shared_rooms",  
                              "calculated_host_listings_count_entire_homes"]
        scaler = StandardScaler()
        self.scaler_pca_host = scaler.fit(self.data[self.host_listings])
        host_listings_df = self.scaler_pca_host.transform(self.data[self.host_listings])
        self.pca_host = PCA(n_components = 3).fit(host_listings_df)
       
        # PCA for minimum nights
        self.min_nights = ["minimum_nights", "minimum_minimum_nights", "maximum_minimum_nights", "minimum_nights_avg_ntm"]
        scaler = StandardScaler()
        self.scaler_pca_min_nights = scaler.fit(self.data[self.min_nights])
        min_nights_df = self.scaler_pca_min_nights.transform(self.data[self.min_nights])
        self.pca_min_nights = PCA(n_components = 1).fit(min_nights_df)

        # PCA for availability
        self.avail = ["availability_365", "availability_30", "availability_60", "availability_90"]
        scaler = StandardScaler()
        self.scaler_pca_avail = scaler.fit(self.data[self.avail])
        avail_df = self.scaler_pca_avail.transform(self.data[self.avail])
        self.pca_avail = PCA(n_components = 1).fit(avail_df)

        # PCA for review total score
        self.review_total_scores = ["review_scores_rating", "mean_compound", "most_pos_compound", "mean_positivity",
                                    "mean_neutrality", "mean_negativity", "most_neg_compound", "prop_of_neg_comp"]
        scaler = StandardScaler()
        self.scaler_pca_review = scaler.fit(self.data[self.review_total_scores])
        review_total_scores_df = self.scaler_pca_review.transform(self.data[self.review_total_scores])
        self.pca_review = PCA(n_components = 4).fit(review_total_scores_df)

        # PCA for maximum nights
        self.max_nights = ["maximum_nights", "minimum_maximum_nights", "maximum_maximum_nights", 
                           "maximum_nights_avg_ntm", "Long term stays allowed"]
        scaler = StandardScaler()
        self.scaler_pca_max_nights = scaler.fit(self.data[self.max_nights])
        max_nights_df = self.scaler_pca_max_nights.transform(self.data[self.max_nights])
        self.pca_max_nights = PCA(n_components = 1).fit(max_nights_df)

        # PCA for amount of reviews
        self.review_amount = ["number_of_reviews_l30d", "number_of_reviews_ltm", "reviews_per_month"]
        scaler = StandardScaler()
        self.scaler_pca_review_amount = scaler.fit(self.data[self.review_amount])
        review_amount_df = self.scaler_pca_review_amount.transform(self.data[self.review_amount])
        self.pca_review_amount = PCA(n_components = 2).fit(review_amount_df)

        # PCA for host about
        self.host_ab = ["compound_host_ab", "positivity_host_ab", "host_about_length", "neutrality_host_ab"]
        scaler = StandardScaler()
        self.scaler_pca_host_ab = scaler.fit(self.data[self.host_ab])
        host_ab_df = self.scaler_pca_host_ab.transform(self.data[self.host_ab])
        self.pca_host_ab = PCA(n_components = 2).fit(host_ab_df)

        # PCA for neighborhood overview
        self.neigh_over = ["compound_neigh_over", "positivity_neigh_over", "neighborhood_overview_length", "neutrality_neigh_over"]
        scaler = StandardScaler()
        self.scaler_pca_neigh_over = scaler.fit(self.data[self.neigh_over])
        neigh_over_df = self.scaler_pca_neigh_over.transform(self.data[self.neigh_over])
        self.pca_neigh_over = PCA(n_components = 2).fit(neigh_over_df)

        # PCA for amount of reviews
        self.descr = ["compound_descr", "positivity_descr", "description_length", "neutrality_descr"]
        scaler = StandardScaler()
        self.scaler_pca_descr = scaler.fit(self.data[self.descr])
        descr_df = self.scaler_pca_descr.transform(self.data[self.descr])
        self.pca_descr = PCA(n_components = 2).fit(descr_df)

        # PCAs for image numbers
        self.img_no = ["no_img_others", "no_img_hallway", "no_img_dining", "no_img_bathroom", "count", 
                       "no_img_bedroom", "no_img_kitchen", "no_img_living"]
        scaler = StandardScaler()
        self.scaler_pca_img_no = scaler.fit(self.data[self.img_no])
        img_no_df = self.scaler_pca_img_no.transform(self.data[self.img_no])
        self.pca_img_no = PCA(n_components = 5).fit(img_no_df)


        return self 
    
    def transform_second(self):
        # PCA TRANSFORMS
        city_life_df = self.scaler_pca_city_life.transform(self.data[self.city_life])
        city_pcas = self.pca_city.transform(city_life_df)
        self.data["city_life_pca1"] = city_pcas[:,0]
        self.data["city_life_pca2"] = city_pcas[:,1]
        self.data["city_life_pca3"] = city_pcas[:,2]
        self.data["city_life_pca4"] = city_pcas[:,3]
        self.data["city_life_pca5"] = city_pcas[:,4]
        self.data = drop_col(self.data, self.city_life, regex = False)
        
        travel_touristic_df = self.scaler_pca_travel.transform(self.data[self.travel_touristic])
        self.data["travel_touristic_pca"] = self.pca_travel.transform(travel_touristic_df)
        self.data = drop_col(self.data, self.travel_touristic, regex = False)

        kitchen_df = self.scaler_pca_kitchen.transform(self.data[self.kitchen])
        kitchen_pcas = self.pca_kitchen.transform(kitchen_df)
        self.data["kitchen_pca1"] = kitchen_pcas[:,0]
        self.data["kitchen_pca2"] = kitchen_pcas[:,1]
        self.data["kitchen_pca3"] = kitchen_pcas[:,2]
        self.data["kitchen_pca4"] = kitchen_pcas[:,3]
        self.data = drop_col(self.data, self.kitchen, regex = False)

        accommodation_size_df = self.scaler_pca_acco.transform(self.data[self.acco])
        acco_size_pcas = self.pca_acco_size.transform(accommodation_size_df)
        self.data["accommodation_size_pca1"] = acco_size_pcas[:, 0]
        self.data["accommodation_size_pca2"] = acco_size_pcas[:, 1]
        self.data["accommodation_size_pca3"] = acco_size_pcas[:, 2]
        self.data["accommodation_size_pca4"] = acco_size_pcas[:, 3]
        self.data["accommodation_size_pca5"] = acco_size_pcas[:, 4]
        self.data["accommodation_size_pca6"] = acco_size_pcas[:, 5]
        self.data = drop_col(self.data, self.acco, regex = False)

        host_listings_df = self.scaler_pca_host.transform(self.data[self.host_listings])
        host_listings_pcas = self.pca_host.transform(host_listings_df)
        self.data["host_listings_pca1"] = host_listings_pcas[:,0]
        self.data["host_listings_pca2"] = host_listings_pcas[:,1]
        self.data["host_listings_pca3"] = host_listings_pcas[:,2]
        self.data = drop_col(self.data, self.host_listings, regex = False)
        
        min_nights_df = self.scaler_pca_min_nights.transform(self.data[self.min_nights])
        self.data["min_nights_pca"] = self.pca_min_nights.transform(min_nights_df)
        self.data = drop_col(self.data, self.min_nights, regex = False)
        
        avail_df = self.scaler_pca_avail.transform(self.data[self.avail])
        self.data["availability_pca"] = self.pca_avail.transform(avail_df)
        self.data = drop_col(self.data, self.avail, regex = False)
        
        review_total_scores_df = self.scaler_pca_review.transform(self.data[self.review_total_scores])
        review_total_pcas = self.pca_review.transform(review_total_scores_df)
        self.data["review_total_pca1"] = review_total_pcas[:, 0]
        self.data["review_total_pca2"] = review_total_pcas[:, 1]
        self.data["review_total_pca3"] = review_total_pcas[:, 2]
        self.data["review_total_pca4"] = review_total_pcas[:, 3]
        self.data = drop_col(self.data, self.review_total_scores, regex = False)
        
        max_nights_df = self.scaler_pca_max_nights.transform(self.data[self.max_nights])
        self.data["max_nights_pca"] = self.pca_max_nights.transform(max_nights_df)
        self.data = drop_col(self.data, self.max_nights, regex = False)
        
        review_amount_df = self.scaler_pca_review_amount.transform(self.data[self.review_amount])
        review_amount_pcas = self.pca_review_amount.transform(review_amount_df)
        self.data["review_amount_pca1"] = review_amount_pcas[:,0]
        self.data["review_amount_pca2"] = review_amount_pcas[:,1]
        self.data = drop_col(self.data, self.review_amount, regex = False)

        host_ab_df = self.scaler_pca_host_ab.transform(self.data[self.host_ab])
        host_ab_pcas = self.pca_host_ab.transform(host_ab_df)
        self.data["host_ab_pca1"] = host_ab_pcas[:,0]
        self.data["host_ab_pca2"] = host_ab_pcas[:,1]
        self.data = drop_col(self.data, self.host_ab, regex = False)

        neigh_over_df = self.scaler_pca_neigh_over.transform(self.data[self.neigh_over])
        neigh_over_pcas = self.pca_neigh_over.transform(neigh_over_df)
        self.data["neigh_over_pca1"] = neigh_over_pcas[:,0]
        self.data["neigh_over_pca2"] = neigh_over_pcas[:,1]
        self.data = drop_col(self.data, self.neigh_over, regex = False)

        descr_df = self.scaler_pca_descr.transform(self.data[self.descr])
        descr_pcas = self.pca_descr.transform(descr_df)
        self.data["descr_pca1"] = descr_pcas[:,0]
        self.data["descr_pca2"] = descr_pcas[:,1]
        self.data = drop_col(self.data, self.descr, regex = False)

        # PCA TRANSFORMS
        img_no_df = self.scaler_pca_img_no.transform(self.data[self.img_no])
        image_pcas = self.pca_img_no.transform(img_no_df)
        self.data["img_no_pca1"] = image_pcas[:,0]
        self.data["img_no_pca2"] = image_pcas[:,1]
        self.data["img_no_pca3"] = image_pcas[:,2]
        self.data["img_no_pca4"] = image_pcas[:,3]
        self.data["img_no_pca5"] = image_pcas[:,4]
        self.data = drop_col(self.data, self.img_no, regex = False)
                
        # DROP DUE TO CORRELATION  
        # keep Dryer available
        self.data = drop_col(self.data, ["Washer_available"], regex = False) 
        # no good PCA, keep Shampoo_Conditioner_available
        self.data = drop_col(self.data, ["Hangers", "Hair dryer", "Iron"], regex = False) 
        # keep Washer available, Kitchen in PCA
        self.data = drop_col(self.data, ["Smoke alarm", "host_location_country_Ireland"], regex = False) 
        # keep fire extinguisher
        self.data = drop_col(self.data, ["First aid kit"], regex = False) 
        # keep Bed linens
        self.data = drop_col(self.data, ["Hot water"], regex = False) 
        # keep Private Entrance
        self.data = drop_col(self.data, ["Cable TV", "Indoor fireplace"], regex = False) 
        # keep Safe_available
        self.data = drop_col(self.data, ["Paid_parking", "Shower gel", "Bathtub", "Baby_friendly",], regex = False) 
        # Dishwasher in KitchenPCA, keep garden_available
        self.data = drop_col(self.data, ["Coffee_machine_available", "Patio_balcony_available"], regex = False) 
        # keep Breakfast, bath private in bath PCA
        self.data = drop_col(self.data, ["Host greets you"], regex = False) 
        # keep last_review
        self.data = drop_col(self.data, ["first_review"], regex = False) 
        # PCA does not work that good, keep "review_scores_communication"
        self.data = drop_col(self.data, ["review_scores_location", "review_scores_accuracy",   
                                        "review_scores_cleanliness", "review_scores_value"], regex = False) 
        # keep breakfast
        self.data = drop_col(self.data, ["Lock on bedroom door"], regex = False) 
        # keep Private Entrance
        self.data = drop_col(self.data, ["Safe_available", "Garden_backyard_available"], regex = False) 
        # will correlate with kitchen pca
        self.data = drop_col(self.data, ["Bed linens"], regex = False) 
        
        print("PCA's built and correlated features dropped.")
        
        return self.data
    
    def fit_third(self):
        
        # T-TESTS
        # get binary variables
        bin_col = [col for col in self.data if (np.isin(self.data[col].unique(), [0, 1]).all() or np.isin(self.data[col].unique(), [0., 1.]).all())]

        stats_val = []
        p_val = []
        names = []

        price = self.data["price"]
        price = price.str.replace("$","")
        price = price.str.replace(",","")
        price = price.astype(float)
        price = np.log(price)

        p = price
        for i in bin_col:
            t_Test(self.data[i], p, stats_val, p_val, names)
        
        p_val_sig = []
        for x in p_val:
            p_val_sig.append(x < 0.05)
        
        insig = [x for x, y in zip(names, p_val_sig) if y == False]
        self.insig = insig
        
        num_col = [col for col in self.data if ~np.isin(self.data[col].unique(), [0, 1]).all()]
        num_col.remove("price")
        num_col.remove("id")
        scaler = StandardScaler()
        self.scaler_final = scaler.fit(self.data[num_col])
        
        return self
    
    def transform_third(self, log_transform = True, drop_id = True, standardize = True):
        # T-TESTS
        if len(self.insig) > 0:
            self.data = self.data.drop(self.insig, axis = 1)
        print("Due to insignificant t-tests we drop:")
        print(self.insig)
        
        price = self.data["price"]
        price = price.str.replace("$","")
        price = price.str.replace(",","")
        price = price.astype(float)
        
        if log_transform:
            price = np.log(price)
        if drop_id:
            self.data.drop("id", axis = 1, inplace = True)
            
        self.data.drop("price", axis = 1, inplace = True)

        num_col = [col for col in self.data if ~np.isin(self.data[col].unique(), [0, 1]).all()]
        if drop_id == False:
            num_col.remove("id")
            
        if standardize == True:
            self.data[num_col] = self.scaler_final.transform(self.data[num_col])
        
        return self.data, price

    def fit_transform(self, X, log_transform = True, drop_id = True):
        print('-'*30)
        print('Fit and Transform data...')
        print('-'*30)
        self.data = X
        self.data = self.preprocess()
        self.data = self.process_amenities(fit = True)
        self.data = self.add_stuff()
        self.fit_first()
        self.data = self.transform_first(fit = True)
        self.fit_second()
        self.data = self.transform_second()
        self.fit_third()
        self.data, price = self.transform_third(log_transform, drop_id)
        self.data.columns = self.data.columns.str.replace(" ","_")       
        return self.data, price
        
    def transform(self, X, log_transform = True, drop_id = True):
        print('-'*30)
        print('Transform data...')
        print('-'*30)
        self.data = X
        self.data = self.preprocess()
        self.data = self.process_amenities(fit = False)
        self.data = self.add_stuff(munich = True)
        self.data = self.transform_first(fit = False, munich = True)
        self.data = self.transform_second()
        self.data, price = self.transform_third(log_transform, drop_id)       
        self.data.columns = self.data.columns.str.replace(" ","_")       
        return self.data, price

    def fit_transform_dendro(self, X, log_transform = True, drop_id = True, standardize = True):
        print('-'*30)
        print('Fit and Transform data...')
        print('-'*30)
        self.data = X
        self.data = self.preprocess()
        self.data = self.process_amenities(fit = True)
        self.data = self.add_stuff()
        self.fit_first()
        self.data = self.transform_first(fit = True)
        self.fit_third()
        self.data, price = self.transform_third(log_transform, drop_id, standardize = standardize)
        self.data.columns = self.data.columns.str.replace(" ","_")       
        return self.data, price
        
    def transform_dendro(self, X, log_transform = True, drop_id = True, standardize = True):
        print('-'*30)
        print('Transform data...')
        print('-'*30)
        self.data = X
        self.data = self.preprocess()
        self.data = self.process_amenities(fit = False)
        self.data = self.add_stuff(munich = True)
        self.data = self.transform_first(fit = False, munich = True)
        self.data, price = self.transform_third(log_transform, drop_id, standardize = standardize)
        self.data.columns = self.data.columns.str.replace(" ","_")       
        return self.data, price
    
    
def load_data_munich(random_seed = 123, test_split = 0.2, val_split = 0.2, for_dendro = False, drop_id = True, standardize = True):
    url_listing = "http://data.insideairbnb.com/ireland/leinster/dublin/2021-11-07/data/listings.csv.gz"
    listings = pd.read_csv(url_listing)
    
    # remove extreme prices
    price = listings["price"]
    price = price.str.replace("$","")
    price = price.str.replace(",","")
    price = price.astype(float)
    filter = price < 500
    listings = listings[filter]
    wrangler = Wrangler()
    
    X_train, X_test = train_test_split(listings, random_state = random_seed, test_size = test_split)
    X_train, X_val = train_test_split(X_train, random_state = random_seed, test_size = val_split)
    
    if for_dendro:
        X_train, y_train = wrangler.fit_transform_dendro(X_train, drop_id=drop_id, standardize=standardize)
        X_train = X_train[['host_identity_verified', 'instant_bookable', 'Bathtub', 'Bed_linens', 'Breakfast', 'Cable_TV', 'Carbon_monoxide_alarm', 'Cooking_basics', 'Dishes_and_silverware', 'Dishwasher', 'Elevator', 'Fire_extinguisher', 'First_aid_kit', 'Hair_dryer', 'Hangers', 'Host_greets_you', 'Hot_water', 'Indoor_fireplace', 'Iron', 'Lock_on_bedroom_door', 'Long_term_stays_allowed', 'Microwave', 'Private_entrance', 'Shower_gel', 'Smoke_alarm', 'Oven_available', 'Stoves_available', 'Refridgerator_available', 'Garden_backyard_available', 'Paid_parking', 'Workspace', 'Shampoo_Conditioner_available', 'Coffee_machine_available', 'Dryer_available', 'Washer_available', 'Patio_balcony_available', 'Kitchen_available', 'Safe_available', 'TV_available', 'Baby_friendly', 'in_city', 'sex_amenities', 'kiosks', 'bath_number_1', 'bath_number_2', 'bedrooms_1', 'bedrooms_2', 'host_location_country_Ireland', 'neighbourhood_cleansed_Dublin_City', 'property_type_Entire_rental_unit', 'property_type_Entire_residential_home', 'property_type_Others', 'room_type_Entire_home/apt', 'room_type_Private_room', 'bath_kind_Normal', 'bath_kind_Private', 'bath_kind_Shared', 'host_since', 'host_listings_count', 'latitude', 'longitude', 'accommodates', 'beds', 'minimum_nights', 'maximum_nights', 'availability_30', 'availability_60', 'availability_90', 'availability_365', 'number_of_reviews', 'number_of_reviews_ltm', 'number_of_reviews_l30d', 'first_review', 'last_review', 'review_scores_rating', 'review_scores_accuracy', 'review_scores_cleanliness', 'review_scores_communication', 'review_scores_location', 'review_scores_value', 'calculated_host_listings_count', 'reviews_per_month', 'minimum_minimum_nights', 'maximum_minimum_nights', 'minimum_maximum_nights', 'maximum_maximum_nights', 'minimum_nights_avg_ntm', 'maximum_nights_avg_ntm', 'calculated_host_listings_count_entire_homes', 'calculated_host_listings_count_private_rooms', 'calculated_host_listings_count_shared_rooms', 'name_length', 'description_length', 'neighborhood_overview_length', 'host_about_length', 'prop_of_eng_reviews', 'mean_compound', 'mean_negativity', 'mean_neutrality', 'mean_positivity', 'mean_review_length', 'prop_of_neg_comp', 'most_neg_compound', 'most_pos_compound', 'compound_descr', 'negativity_descr', 'neutrality_descr', 'positivity_descr', 'compound_neigh_over', 'negativity_neigh_over', 'neutrality_neigh_over', 'positivity_neigh_over', 'compound_host_ab', 'negativity_host_ab', 'neutrality_host_ab', 'positivity_host_ab', 'bars', 'cafes', 'fast_foods', 'restaurants', 'library', 'university', 'bus_train_tram_station', 'bicycle_rentals', 'parking', 'taxi', 'casinos', 'cinemas', 'nightclubs', 'social_amenities', 'theatres_artscentre', 'malls', 'supermarkets', 'nearest_sight', '2nd_nearest_sight', '3rd_nearest_sight', 'mean_dist_sight', 'nearest_travel_poss', 'mean_dist_travel', 'count', 'brightness', 'contrast', 'no_img_bathroom', 'no_img_bedroom', 'no_img_dining', 'no_img_hallway', 'no_img_kitchen', 'no_img_living', 'no_img_others']]
    else :
        X_train, y_train = wrangler.fit_transform(X_train, drop_id=drop_id)
        X_train = X_train[['host_identity_verified', 'instant_bookable', 'Breakfast', 'Carbon_monoxide_alarm', 'Elevator', 'Fire_extinguisher', 'Private_entrance', 'Workspace', 'Shampoo_Conditioner_available', 'Dryer_available', 'TV_available', 'kiosks', 'host_since', 'latitude', 'longitude', 'number_of_reviews', 'last_review', 'review_scores_communication', 'name_length', 'prop_of_eng_reviews', 'mean_review_length', 'negativity_descr', 'negativity_neigh_over', 'negativity_host_ab', 'parking', 'brightness', 'contrast', 'city_life_pca1', 'city_life_pca2', 'city_life_pca3', 'city_life_pca4', 'city_life_pca5', 'travel_touristic_pca', 'kitchen_pca1', 'kitchen_pca2', 'kitchen_pca3', 'kitchen_pca4', 'accommodation_size_pca1', 'accommodation_size_pca2', 'accommodation_size_pca3', 'accommodation_size_pca4', 'accommodation_size_pca5', 'accommodation_size_pca6', 'host_listings_pca1', 'host_listings_pca2', 'host_listings_pca3', 'min_nights_pca', 'availability_pca', 'review_total_pca1', 'review_total_pca2', 'review_total_pca3', 'review_total_pca4', 'max_nights_pca', 'review_amount_pca1', 'review_amount_pca2', 'host_ab_pca1', 'host_ab_pca2', 'neigh_over_pca1', 'neigh_over_pca2', 'descr_pca1', 'descr_pca2', 'img_no_pca1', 'img_no_pca2', 'img_no_pca3', 'img_no_pca4', 'img_no_pca5']]
    url_listing = "http://data.insideairbnb.com/germany/bv/munich/2021-12-24/data/listings.csv.gz"
    listing_munich = pd.read_csv(url_listing)

    # remove extreme prices
    price = listing_munich["price"]
    price = price.str.replace("$","")
    price = price.str.replace(",","")
    price = price.astype(float)
    filter = price < 500
    listing_munich = listing_munich[filter]
    
    if for_dendro:
        X_munich, y_munich = wrangler.transform_dendro(listing_munich, drop_id=drop_id, standardize=standardize)
        X_munich = X_munich[['host_identity_verified', 'instant_bookable', 'Bathtub', 'Bed_linens', 'Breakfast', 'Cable_TV', 'Carbon_monoxide_alarm', 'Cooking_basics', 'Dishes_and_silverware', 'Dishwasher', 'Elevator', 'Fire_extinguisher', 'First_aid_kit', 'Hair_dryer', 'Hangers', 'Host_greets_you', 'Hot_water', 'Indoor_fireplace', 'Iron', 'Lock_on_bedroom_door', 'Long_term_stays_allowed', 'Microwave', 'Private_entrance', 'Shower_gel', 'Smoke_alarm', 'Oven_available', 'Stoves_available', 'Refridgerator_available', 'Garden_backyard_available', 'Paid_parking', 'Workspace', 'Shampoo_Conditioner_available', 'Coffee_machine_available', 'Dryer_available', 'Washer_available', 'Patio_balcony_available', 'Kitchen_available', 'Safe_available', 'TV_available', 'Baby_friendly', 'in_city', 'sex_amenities', 'kiosks', 'bath_number_1', 'bath_number_2', 'bedrooms_1', 'bedrooms_2', 'host_location_country_Ireland', 'neighbourhood_cleansed_Dublin_City', 'property_type_Entire_rental_unit', 'property_type_Entire_residential_home', 'property_type_Others', 'room_type_Entire_home/apt', 'room_type_Private_room', 'bath_kind_Normal', 'bath_kind_Private', 'bath_kind_Shared', 'host_since', 'host_listings_count', 'latitude', 'longitude', 'accommodates', 'beds', 'minimum_nights', 'maximum_nights', 'availability_30', 'availability_60', 'availability_90', 'availability_365', 'number_of_reviews', 'number_of_reviews_ltm', 'number_of_reviews_l30d', 'first_review', 'last_review', 'review_scores_rating', 'review_scores_accuracy', 'review_scores_cleanliness', 'review_scores_communication', 'review_scores_location', 'review_scores_value', 'calculated_host_listings_count', 'reviews_per_month', 'minimum_minimum_nights', 'maximum_minimum_nights', 'minimum_maximum_nights', 'maximum_maximum_nights', 'minimum_nights_avg_ntm', 'maximum_nights_avg_ntm', 'calculated_host_listings_count_entire_homes', 'calculated_host_listings_count_private_rooms', 'calculated_host_listings_count_shared_rooms', 'name_length', 'description_length', 'neighborhood_overview_length', 'host_about_length', 'prop_of_eng_reviews', 'mean_compound', 'mean_negativity', 'mean_neutrality', 'mean_positivity', 'mean_review_length', 'prop_of_neg_comp', 'most_neg_compound', 'most_pos_compound', 'compound_descr', 'negativity_descr', 'neutrality_descr', 'positivity_descr', 'compound_neigh_over', 'negativity_neigh_over', 'neutrality_neigh_over', 'positivity_neigh_over', 'compound_host_ab', 'negativity_host_ab', 'neutrality_host_ab', 'positivity_host_ab', 'bars', 'cafes', 'fast_foods', 'restaurants', 'library', 'university', 'bus_train_tram_station', 'bicycle_rentals', 'parking', 'taxi', 'casinos', 'cinemas', 'nightclubs', 'social_amenities', 'theatres_artscentre', 'malls', 'supermarkets', 'nearest_sight', '2nd_nearest_sight', '3rd_nearest_sight', 'mean_dist_sight', 'nearest_travel_poss', 'mean_dist_travel', 'count', 'brightness', 'contrast', 'no_img_bathroom', 'no_img_bedroom', 'no_img_dining', 'no_img_hallway', 'no_img_kitchen', 'no_img_living', 'no_img_others']]
    else :
         X_munich, y_munich = wrangler.transform(listing_munich, drop_id=drop_id)
         X_munich = X_munich[['host_identity_verified', 'instant_bookable', 'Breakfast', 'Carbon_monoxide_alarm', 'Elevator', 'Fire_extinguisher', 'Private_entrance', 'Workspace', 'Shampoo_Conditioner_available', 'Dryer_available', 'TV_available', 'kiosks', 'host_since', 'latitude', 'longitude', 'number_of_reviews', 'last_review', 'review_scores_communication', 'name_length', 'prop_of_eng_reviews', 'mean_review_length', 'negativity_descr', 'negativity_neigh_over', 'negativity_host_ab', 'parking', 'brightness', 'contrast', 'city_life_pca1', 'city_life_pca2', 'city_life_pca3', 'city_life_pca4', 'city_life_pca5', 'travel_touristic_pca', 'kitchen_pca1', 'kitchen_pca2', 'kitchen_pca3', 'kitchen_pca4', 'accommodation_size_pca1', 'accommodation_size_pca2', 'accommodation_size_pca3', 'accommodation_size_pca4', 'accommodation_size_pca5', 'accommodation_size_pca6', 'host_listings_pca1', 'host_listings_pca2', 'host_listings_pca3', 'min_nights_pca', 'availability_pca', 'review_total_pca1', 'review_total_pca2', 'review_total_pca3', 'review_total_pca4', 'max_nights_pca', 'review_amount_pca1', 'review_amount_pca2', 'host_ab_pca1', 'host_ab_pca2', 'neigh_over_pca1', 'neigh_over_pca2', 'descr_pca1', 'descr_pca2', 'img_no_pca1', 'img_no_pca2', 'img_no_pca3', 'img_no_pca4', 'img_no_pca5']]


    # transform munich prices to dublin level
    mean_dub = y_train.mean()
    std_dub = y_train.std()
    
    mean_mun = y_munich.mean()
    std_mun = y_munich.std()
    
    y_munich_tmp = (y_munich - mean_mun)/std_mun
    y_munich = (y_munich_tmp * std_dub)+mean_dub

    return X_train, X_munich, y_train, y_munich
