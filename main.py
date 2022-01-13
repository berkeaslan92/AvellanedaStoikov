from AvellanedaStoikov import AvellanedaStoikov

if __name__ == '__main__':

    """
    Execute here
    """

    AS = AvellanedaStoikov(S0=20,
                           T=6.5*3600,
                           model='Trivial')
    AS.setTrivialDeltaValue(0.05)
    AS.execute()
    AS.visualize()
