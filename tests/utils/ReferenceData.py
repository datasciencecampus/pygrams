import pandas as pd

generated_date_year = 1900

cold_abstracts = [
    ('1', '''A refrigerator including a main body provided with a refrigerating chamber at an upper
        section and with a freezing chamber at a lower section, an ice making tray disposed in an upper space 
        of an ice making chamber ined in the refrigerating chamber, a first storage container disposed in a 
        lower space of the ice making chamber to store ice falling down from the ice making tray, and a second 
        storage container disposed in a freezing chamber to store ice transferred from the ice making tray. The 
        main body includes a guide channel to guide, when the first storage container reaches an ice-full 
        state, ice falling from the ice making tray to the second storage container in the freezing chamber. 
        The size of the ice making chamber is greatly reduced while a sufficient amount of the ice may be 
        stored, thus securing a larger available space in the refrigerating chamber.'''),
    ('2', '''The invention provides methods and compositions for maintaining a refrigerating chamber at a 
        constant temperature. This will maintain a quantity of ice stored ready for consumption.'''),
    ('3', '''An ice piece release system that includes a chilled compartment set at a temperature below 
        0.degree. C., a warm section at a temperature above 0.degree. C., and a tray in thermal communication 
        with the chilled compartment. The tray includes a plurality of ice piece-forming receptacles and a 
        cavity in thermal communication with the receptacles. The ice piece release system also includes a 
        primary reservoir assembly in thermal communication with the warm section and fluid communication with 
        the cavity of the tray. The ice piece release system further includes a heat-exchanging fluid having a 
        freezing point below that of water, and the fluid resides in the primary reservoir assembly and the
        cavity of the tray. The primary reservoir assembly is further adapted to move at least a portion of the
        heat-exchanging fluid in the reservoir assembly into the cavity.'''),
    ('4', '''A refrigerator, in particular a household refrigerator, includes an utility chamber for cooled 
        goods and a control device, with which a cold air flow can be introduced into the utility chamber when
        a cooling signal is present. A defrost heating element is rendered operative by the control device to 
        prevent the formation of condensate and/or ice due to the cold air flow fed into the utility chamber. 
        A timing element keeps the heating element out of operation for a predetermined time interval in 
        response to the generation of the cooling signal.''')
]

random_abstracts = [
    ('101', '''Acoustic volume indicators for determining liquid or gas volume within a container comprise a 
        contactor to vibrate a container wall, a detector to receive vibration data from the container wall, 
        a processor to convert vibration data to frequency information and compare the frequency information to 
        characteristic container frequency vs. volume data to obtain the measured volume, and an indicator for 
        displaying the measured volume. The processor may comprise a microprocessor disposed within a housing 
        having lights that each represent a particular volume. The microprocessor is calibrated to provide an 
        output signal to a light that indicates the container volume. The processor may comprise a computer and 
        computer program that converts the data to frequency information, analyzes the frequency information to 
        identify a peak frequency, compares the peak frequency to the characteristic frequency vs. volume data 
        to determine the measured volume, and displays the measured volume on a video monitor. '''),
    ('102', '''A single-module deployable bolted flange connection apparatus ( 10 ) makes up standard flange 
        joints ( 24, 32 ) for various pipeline tie-in situations, such as spool piece connection and 
        flowline-tree connections, without the use of divers and auxiliary multiple pieces of equipment. An 
        outer Flange Alignment Frame (FAF) ( 14 ), carries one or more claws ( 38 ) for grabbing the pipe/spool
        to provide flange alignment. The claws are suspended and driven by a novel arrangement of five 
        hydraulic rams ( 412 - 420 ) A crash-resistant inner frame ( 148 ) houses complete connection tooling 
        ( 150, 152  etc.) The tooling performs the final alignment steps, inserts the gasket and studs, applies 
        the required tension, and connects the nuts. Studs and nuts are stored separately from the tooling in
        an indexed carou, to permit multiple operations, reverse operations (disconnection), and re-work of 
        failed steps, all without external intervention. '''),
    ('103', '''A passenger seat with increased knee space for an aft-seated passenger, including a seat base for 
        being attached to a supporting deck and at least one seat frame including a seat back and seat bottom
        carried by the seat base. At least one arm rest assembly is carried by the seat frame and including an
        arm rest mounted for pivotal movement about a pivot member between a use position with an upper support
        surface in a horizontal position for supporting a forearm of a passenger seated in the seat, and a 
        stowed position wherein the upper support surface of the arm rest is perpendicular to the use position. 
        The arm rest pivot member is mounted on the seat frame at a point forward of a plane ined by the seat 
        back carried by the seat and above a point ined by the seat bottom for allowing the knee of an 
        aft-seated passenger to occupy space behind the pivot member of the arm rest. '''),
    ('104', '''A bag having two accesses, an inlet and an outlet, both containing closing devices for releasing 
        or stopping the flow of a liquid that flows into or out of the bag. The inlet has a filtering element 
        for retaining particles possibly produced by the coring phenomenon which can occur when the spike of 
        the inlet ruptures the plug of the bottle. Also provided is a safety device used for permanently 
        attaching the bottle to the inlet. '''),
    ('105', '''An x-ray tube assembly is provided comprising a tube casing assembly including a plurality of 
        vertical mount posts. An insulator plate is mounted to the plurality of vertical mount posts such that 
        the insulator plate can translate vertically on the posts. A cathode assembly is mounted to the 
        insulator plate and generates both an eccentric moment and a vertical expansion in response to a 
        cathode power load. A semi-compressible element is positioned between at least one of the vertical 
        mount posts and the insulator plate. The semi-compressible element becomes incompressible at a cathode 
        power threshold such that the vertical expansion is translated into a correction moment countering the 
        eccentric moment. ''')
]


def patents_to_df(_patents, _generated_date_year):
    frames = []
    for patent in _patents:
        df = pd.DataFrame(
            {'patent_id': patent[0],
             'application_id': "blah",
             'related_document_ids': [[1, 2]],
             'abstract': patent[1],
             'inventor_names': [[3, 4]],
             'inventor_countries': [[5, 6]],
             'inventor_cities': [[7]],
             'invention_title': "blah",
             'claim1': "blah",
             'classifications_cpc': [[8]],
             'applicant_cities': [[9]],
             'applicant_countries': [[12]],
             'applicant_organisation': [[14]],
             'application_date': "blah",
             'publication_date': pd.Timestamp(f'{_generated_date_year}-11-04 00:00:00'),
             'patents_cited': [[16]]
             })
        _generated_date_year += 1
        frames.append(df)

    combined_df = pd.concat(frames)
    combined_df.set_index('patent_id', inplace=True, drop=False, verify_integrity=True)
    return combined_df


cold_df = patents_to_df(cold_abstracts, generated_date_year)
random_df = patents_to_df(random_abstracts, generated_date_year)
