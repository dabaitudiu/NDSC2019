import pickle

def initialize():

    # load benefit class 1
    benefit_1 = pickle.load(open("benefit1.pkl","rb"))
    print("-"*30,"benefit_1 load finished.","-"*30)

    # load benefit class 2
    2_a = pickle.load(open("benefit2-a.pkl","rb"))
    2_b = pickle.load(open("benefit2-b.pkl","rb"))
    2_c = pickle.load(open("benefit2-c.pkl","rb"))
    benefit_2 = 2_a + 2_b + 2_c
    print("-"*30,"benefit_2 load finished.","-"*30)


    # load benefit class 3
    benefit_3 = pickle.load(open("benefit3.pkl","rb"))
    print("-"*30,"benefit_3 load finished.","-"*30)

    # load benefit class 4
    benefit_4 = pickle.load(open("benefit4.pkl","rb"))
    print("-"*30,"benefit_4 load finished.","-"*30)

    # load benefit class 5
    5_a = pickle.load(open("benefit5-a.pkl","rb"))
    5_b = pickle.load(open("benefit5-b.pkl","rb"))
    benefit_5 = 5_a + 5_b
    print("-"*30,"benefit_5 load finished.","-"*30)

    # load benefit class 6
    benefit_6 = pickle.load(open("benefit6.pkl","rb"))
    print("-"*30,"benefit_6 load finished.","-"*30)

    return benefit_1,benefit_2,benefit_3,benefit_4,benefit_5,benefit_6
