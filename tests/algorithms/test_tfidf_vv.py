import statistics
import unittest

import pandas as pd

from scripts import FilePaths
from scripts.algorithms.tfidf import StemTokenizer, TFIDF
from tests.utils.DiceScore import Dice
from tests.vvcode.abstracts2pickle import us_vv_patents_pickle_name


# Sample abstracts taken from the USPTO Bulk Download Service: https://bulkdata.uspto.gov
# Data used was downloaded from "Patent Grant Full Text Data"


class TestsSetup(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.common_tokenizer = StemTokenizer()
        cls.dice_threshold = 0.5
        cls.total_dice_n = 0.0
        cls.total_dice_u = 0.0
        cls.total_dice_bi = 0.0
        cls.n_tests = 0

        cls.dice_n = []
        cls.dice_u = []
        cls.dice_bi = []

        df = pd.read_pickle(FilePaths.us_patents_random_10000_pickle_name)
        test_dataframe = pd.read_pickle(us_vv_patents_pickle_name)

        df.append(test_dataframe)
        cls.tfidf = TFIDF(df, ngram_range=(1, 3), max_document_frequency=0.3, tokenizer=StemTokenizer())


# noinspection PyPep8Naming
@unittest.skip("these are not unit tests, but a means to optimize tf-idf parameters")
class VV_TF_IDF_Tests(TestsSetup):

    @classmethod
    def setUpClass(cls):
        super(VV_TF_IDF_Tests, cls).setUpClass()
        cls.total_dice_n = 0
        cls.total_dice_u = 0
        cls.total_dice_bi = 0
        cls.dice_n = []
        cls.dice_u = []
        cls.dice_bi = []
        cls.n_tests = 0

    def assertGreaterOrEqualDiceScore(self, expected_terms, actual_termsin):
        dice = Dice(expected_terms, tokenizer=self.common_tokenizer)
        actual_terms = actual_termsin[0]

        dice_score_n, matched_n, TP_n, FN_n, FP_n = dice.get_score_ngrams(actual_terms)
        VV_TF_IDF_Tests.total_dice_n += dice_score_n

        dice_score_u, actual_u, TP_u, FN_u, FP_u = dice.get_score_unigrams(actual_terms)
        VV_TF_IDF_Tests.total_dice_u += dice_score_u

        dice_score_bi, actual_bi, TP_bi, FN_bi, FP_bi = dice.get_score_bigrams(actual_terms)
        VV_TF_IDF_Tests.total_dice_bi += dice_score_bi

        VV_TF_IDF_Tests.dice_n.append(dice_score_n)
        VV_TF_IDF_Tests.dice_u.append(dice_score_u)
        VV_TF_IDF_Tests.dice_bi.append(dice_score_bi)

        if VV_TF_IDF_Tests.n_tests > 1:
            print(
                f" dice_n:  avg={statistics.mean(VV_TF_IDF_Tests.dice_n):0.3},"
                f" std={statistics.stdev(VV_TF_IDF_Tests.dice_n):0.3}")
            print(
                f" dice_u:  avg={statistics.mean(VV_TF_IDF_Tests.dice_u):0.3},"
                f" std={statistics.stdev(VV_TF_IDF_Tests.dice_u):0.3}")
            print(
                f" dice_bi: avg={statistics.mean(VV_TF_IDF_Tests.dice_bi):0.3},"
                f" std={statistics.stdev(VV_TF_IDF_Tests.dice_bi):0.3}")

        # shall we do try as well?
        VV_TF_IDF_Tests.n_tests += 1
        if dice_score_u < self.dice_threshold:
            tokenised_expected_terms_n = dice.expected_token_ngrams
            tokenised_expected_terms_u = dice.expected_token_unigrams
            tokenised_expected_terms_bi = dice.expected_token_bigrams

            self.fail(
                f'\n===================N-GRAMS============================\n'
                f'expected: {tokenised_expected_terms_n} \n'
                f'actual: {actual_terms} \n'
                f'TP: {TP_n} \n'
                f'FP: {FP_n} \n'
                f'FN: {FN_n} \n'
                f'(dice_score_n = {dice_score_n:0.2} : {self.dice_threshold})\n'
                f'==================UNIGRAMS=============================\n'
                f'expected: {tokenised_expected_terms_u} \n'
                f'actual: {actual_u} \n'
                f'TP: {TP_u} \n'
                f'FP: {FP_u} \n'
                f'FN: {FN_u} \n'
                f'(dice_score_n = {dice_score_u:0.2} : {self.dice_threshold})\n'
                f'===================BIGRAMS============================\n'
                f'expected: {tokenised_expected_terms_bi} \n'
                f'actual: {actual_bi} \n'
                f'TP: {TP_bi} \n'
                f'FP: {FP_bi} \n'
                f'FN: {FN_bi} \n'
                f'(dice_score_n = {dice_score_bi:0.2} : {self.dice_threshold})\n'
                f'===============================================\n'
            )

    # #############################V&V FRAMEWORK ##################################################

    def test_patent_(self):
        text = '''A refrigerator including a main body provided with a refrigerating chamber at an upper
            section and with a freezing chamber at a lower section, an ice making tray disposed in an upper space of an
            ice making chamber defined in the refrigerating chamber, a first storage container disposed in a lower space
            of the ice making chamber to store ice falling down from the ice making tray, and a second storage container
            disposed in a freezing chamber to store ice transferred from the ice making tray. The main body includes a
            guide channel to guide, when the first storage container reaches an ice-full state, ice falling from the ice
            making tray to the second storage container in the freezing chamber. The size of the ice making chamber is
            greatly reduced while a sufficient amount of the ice may be stored, thus securing a larger available space
            in the refrigerating chamber.'''

        expected = ['refrigerator', 'freezing chamber', 'ice making chamber', 'guide channel', 'refrigerating chamber']
        actual = self.tfidf.extract_popular_ngrams(text)

        self.assertGreaterOrEqualDiceScore(expected, actual)

    def test_patent_US07229649_20070612(self):
        text = '''The invention provides methods and compositions for maintaining a state of wellness in a human by 
        providing a dietary supplement comprising L-arginine, alone or in combination with ginseng and ginkgo biloba 
        and/or additional nutritional supplements. The invention provides a unique blend of components that, 
        in combination, synergistically bestow sexual wellness upon a human when taken regularly as a dietary 
        supplement. '''
        expected = ['dietary', 'supplement', 'L-arginine', 'ginseng', 'ginkgo', 'biloba', 'nutritional', 'sexual']
        actual = self.tfidf.extract_popular_ngrams(text)

        print(
            "test_patent_US07229649_20070612: We lost 'ginseng', 'ginkgo', 'biloba' | are we stopping non-english "
            "dictionary terms? ")

        self.assertGreaterOrEqualDiceScore(expected, actual)

    def test_patent_US07564214_20090721(self):
        text = '''A battery pack having a secondary battery and a circuit that controls charging and discharging 
        of the secondary battery is provided. The battery pack has a computer that communicates with another computer 
        disposed in a charging device, authenticate the charging device, and calculates remaining capacity information 
        of the secondary battery. When the computer has not successfully authenticated the charging device and has 
        detected that the secondary battery has been charged in a predetermined manner, 
        the computer forcibly sets the remaining capacity information to “no remaining capacity”.'''
        expected = ['battery', 'pack', 'computer', 'charging', 'secondary', 'capacity', 'secondary battery']
        actual = self.tfidf.extract_popular_ngrams(text)

        self.assertGreaterOrEqualDiceScore(expected, actual)

    def test_patent_US09406049_20160802(self):
        text = '''A method, device and system are provided for automated updating of message recipients designated for 
        a reply message based on a declaration or notification in a body of the message. Once a reply message is 
        initiated at a first device and an initial recipient set defined for the reply message based on the participants
        of a previous message of the thread, a declaration or instruction may be inserted in the message body 
        identifying a change to be made to the initial recipient set. In response to a trigger detected at the device, 
        any such declarations are identified and a determination is made whether the current recipient set is consistent
        with them. If not, changes are made to the current recipient set to render it consistent.'''
        expected = ['message', 'recipients', 'reply', 'participants', 'thread', 'device']
        actual = self.tfidf.extract_popular_ngrams(text)

        self.assertGreaterOrEqualDiceScore(expected, actual)

    def test_patent_US09586496_20170307(self):
        text = '''A modular stacked DC architecture for traction system includes a propulsion system includes an 
        electric drive, a direct current (DC) link electrically coupled to the electric drive, and a first DC-DC 
        converter coupled to the DC link. A first energy storage device (ESD) is electrically coupled to the first DC-DC
        converter, and a second DC-DC converter is coupled to the DC link and to the first DC-DC converter. 
        The system also includes a second energy storage device electrically coupled to the second DC-DC converter and a
        controller coupled to the first and second DC-DC converters and configured to control a transfer of energy 
        between the first ESD and the DC link via the first and second DC-DC converters.'''
        expected = ['DC', 'architecture', 'electric', 'drive', 'energy', 'storage', 'device', 'coupled', 'converter',
                    'link', 'DC-DC', 'ESD', 'energy storage', 'electric drive', 'storage device']
        actual = self.tfidf.extract_popular_ngrams(text)

        self.assertGreaterOrEqualDiceScore(expected, actual)

    def test_patent_US08249168_20120821(self):
        text = '''A system and method are disclosed for providing improved processing of video data. A multi-instance 
        encoding module receives combined video and audio input, which is then separated into a video and audio 
        source streams. The video source stream is pre-processed and corresponding video encoder instances are 
        initiated. The preprocessed video source stream is split into video data components, which are assigned to a 
        corresponding encoder instance. Encoding operations are performed by each video encoder instance to generate 
        video output components. The video output components are then assembled in a predetermined sequence to 
        generate an encoded video output stream. Concurrently, the audio source stream is encoded with an audio 
        encoder to generate an encoded audio output stream. The encoded video and audio output streams are combined 
        to generate a combined encoded output stream, which is provided as combined video and audio output. '''
        expected = ['video', 'data', 'encoding', 'audio', 'streams', 'input', 'output', 'source', 'video data',
                    'audio input', 'audio source', 'encoded video', 'output stream', 'video output', 'audio output']
        actual = self.tfidf.extract_popular_ngrams(text)

        self.assertGreaterOrEqualDiceScore(expected, actual)

    def test_patent_US07590927_20090915(self):
        text = '''Outputting information for recovering a sequence of data is disclosed. 
        Outputting includes making a decision that selects a first sequence of states corresponding to a surviving path, 
        determining a second sequence of states corresponding to a non-surviving path associated with the decision, 
        and defining a possible error event based at least in part on the second sequence of states.'''
        expected = ['recovering', 'sequence', 'data', 'state']
        actual = self.tfidf.extract_popular_ngrams(text)

        self.assertGreaterOrEqualDiceScore(expected, actual)

    def test_patent_US07946168_20110524(self):
        text = '''Acoustic volume indicators for determining liquid or gas volume within a container comprise a 
        contactor to vibrate a container wall, a detector to receive vibration data from the container wall, 
        a processor to convert vibration data to frequency information and compare the frequency information to 
        characteristic container frequency vs. volume data to obtain the measured volume, and an indicator for 
        displaying the measured volume. The processor may comprise a microprocessor disposed within a housing having 
        lights that each represent a particular volume. The microprocessor is calibrated to provide an output signal 
        to a light that indicates the container volume. The processor may comprise a computer and computer program 
        that converts the data to frequency information, analyzes the frequency information to identify a peak 
        frequency, compares the peak frequency to the characteristic frequency vs. volume data to determine the 
        measured volume, and displays the measured volume on a video monitor. '''
        expected = ['acoustic', 'liquid', 'gas volume', 'gas', 'container', 'vibration data', 'frequency',
                    'frequency information', 'microprocessor',
                    'housing', 'lights', 'volume data', 'processor', 'calibrated', 'computer', 'computer program',
                    'peak frequency', 'characteristic frequency', 'video monitor']
        actual = self.tfidf.extract_popular_ngrams(text)

        self.assertGreaterOrEqualDiceScore(expected, actual)

    def test_patent_US07445404_20081104(self):
        text = '''A single-module deployable bolted flange connection apparatus ( 10 ) makes up standard flange 
        joints ( 24, 32 ) for various pipeline tie-in situations, such as spool piece connection and flowline-tree 
        connections, without the use of divers and auxiliary multiple pieces of equipment. An outer Flange Alignment 
        Frame (FAF) ( 14 ), carries one or more claws ( 38 ) for grabbing the pipe/spool to provide flange alignment. 
        The claws are suspended and driven by a novel arrangement of five hydraulic rams ( 412 - 420 ) A 
        crash-resistant inner frame ( 148 ) houses complete connection tooling ( 150, 152  etc.) The tooling performs 
        the final alignment steps, inserts the gasket and studs, applies the required tension, and connects the nuts. 
        Studs and nuts are stored separately from the tooling in an indexed carousel, to permit multiple operations, 
        reverse operations (disconnection), and re-work of failed steps, all without external intervention. '''

        expected = ['bolted', 'flange', 'spool', 'nuts' 'claws', 'pipe', 'hydraulic', 'rams', 'gasket', 'studs',
                    'bolted flange', 'flange joints', 'flange alignment', 'hydraulic rams', 'indexed carousel']
        actual = self.tfidf.extract_popular_ngrams(text)

        self.assertGreaterOrEqualDiceScore(expected, actual)

    def test_patent_US08800128_20140812(self):
        text = '''A workpiece clamping jig configured to clamp a workpiece is provided with a baseplate, 
        a fixed abutting member, and a movable abutting member. The baseplate has a pocket, which forms an air intake 
        passage on the reverse side of a workpiece mounting surface, and a suction through-hole internally connecting 
        the workpiece mounting surface and the pocket. The fixed abutting member is disposed on the baseplate and has 
        a fixed abutting surface corresponding to the shape of the workpiece, and the movable abutting member has an 
        abutting surface movable relative to the workpiece caused to abut against the fixed abutting surface of the 
        fixed abutting member. '''
        expected = ['clamping', 'jig', 'suction', 'baseplate', 'abutting member', 'clamping jig', 'abutting surface']
        actual = self.tfidf.extract_popular_ngrams(text)

        self.assertGreaterOrEqualDiceScore(expected, actual)

    def test_patent_US09376276_20160628(self):
        text = '''A sheet feeding apparatus includes a containing unit, a lifting and lowering unit, a loosening 
        unit, a feed unit, a sheet-surface detection unit, an aligning member, a trailing-edge detection unit, 
        a double-feed detection unit, and a control unit. Sheets, of a sheet bundle contained in the containing unit 
        and lifted and lowered, are loosened by blowing air into the sheet bundle and fed. An end face of the 
        contained sheet bundle is aligned on a trailing-edge side. The trailing-edge detection unit moves with the 
        aligning member and detects a sheet surface of a top of the sheet bundle on the trailing-edge side. Where 
        double feed from the containing unit is detected, a detection operation detects, in consideration of a 
        trailing-edge detection unit result, whether the aligning member is located at a position at which the end 
        face of the sheet bundle on the trailing-edge side is aligned. '''
        expected = ['detection', 'sheet', 'sheet feeding', 'detection unit']
        actual = self.tfidf.extract_popular_ngrams(text)

        self.assertGreaterOrEqualDiceScore(expected, actual)

    def test_patent_US08829872_20140909(self):
        text = '''Methods and systems for dropping and/or adding phases in multiphase regulators according to various 
        aspects of the present invention may operate in conjunction with multiple output circuits configured to 
        deliver power to a load and a controller. The controller may be connected to each of the output circuits, 
        such as to drive the output circuits. The controller may be adapted to selectively disable and/or enable 
        phases. For example, the controller may disable one output circuit without disabling another output circuit. 
        In addition, the controller may smoothly reduce the power delivered to the load by the output circuit prior 
        to disabling it, for example to control output glitches. '''
        expected = ['phases', 'regulators', 'multiphase regulators']
        actual = self.tfidf.extract_popular_ngrams(text)

        self.assertGreaterOrEqualDiceScore(expected, actual)

    def test_patent_US06899397_20050531(self):
        text = '''A passenger seat with increased knee space for an aft-seated passenger, including a seat base for 
        being attached to a supporting deck and at least one seat frame including a seat back and seat bottom carried 
        by the seat base. At least one arm rest assembly is carried by the seat frame and including an arm rest 
        mounted for pivotal movement about a pivot member between a use position with an upper support surface in a 
        horizontal position for supporting a forearm of a passenger seated in the seat, and a stowed position wherein 
        the upper support surface of the arm rest is perpendicular to the use position. The arm rest pivot member is 
        mounted on the seat frame at a point forward of a plane defined by the seat back carried by the seat and 
        above a point defined by the seat bottom for allowing the knee of an aft-seated passenger to occupy space 
        behind the pivot member of the arm rest. '''
        expected = ['pivot', 'passenger seat', 'knee space', 'seat base', 'arm rest', 'pivotal movement',
                    'pivot member', 'seat frame']
        actual = self.tfidf.extract_popular_ngrams(text)

        self.assertGreaterOrEqualDiceScore(expected, actual)

    def test_patent_US09833381_20171205(self):
        text = '''A bag having two accesses, an inlet and an outlet, both containing closing devices for releasing or 
        stopping the flow of a liquid that flows into or out of the bag. The inlet has a filtering element for 
        retaining particles possibly produced by the coring phenomenon which can occur when the spike of the inlet 
        ruptures the plug of the bottle. Also provided is a safety device used for permanently attaching the bottle 
        to the inlet. '''
        expected = ['bag', 'closing devices', 'release', 'flow', 'filtering element', 'particles', 'coring phenomenon',
                    'spike', 'ruptures', 'plug', 'inlet', 'safety device', 'liquid flow']
        actual = self.tfidf.extract_popular_ngrams(text)

        self.assertGreaterOrEqualDiceScore(expected, actual)

    def test_patent_US07450690_20081111(self):
        text = '''An x-ray tube assembly is provided comprising a tube casing assembly including a plurality of 
        vertical mount posts. An insulator plate is mounted to the plurality of vertical mount posts such that the 
        insulator plate can translate vertically on the posts. A cathode assembly is mounted to the insulator plate 
        and generates both an eccentric moment and a vertical expansion in response to a cathode power load. A 
        semi-compressible element is positioned between at least one of the vertical mount posts and the insulator 
        plate. The semi-compressible element becomes incompressible at a cathode power threshold such that the 
        vertical expansion is translated into a correction moment countering the eccentric moment. '''
        expected = ['x-ray tube', 'x-ray', 'tube', 'tube casing assembly', 'vertical mount', 'insulator plate',
                    'cathode', 'eccentric moment', 'semi-compressible element',
                    'incompressible', 'vertical expansion', 'eccentric moment']
        actual = self.tfidf.extract_popular_ngrams(text)

        self.assertGreaterOrEqualDiceScore(expected, actual)

    def test_patent_US08597953_20131203(self):
        text = '''A set of volatile organic compounds is provided, comprising at least butylated hydroxy toluene or 
        4,6-di (1,1-dimethylethyl)_2-methyl-phenol for breath analysis. Methods of use thereof in diagnosing, 
        monitoring or prognosing lung cancer are also disclosed. '''
        expected = ['organic compounds', 'butylated hydroxy', 'toluene' 'breath analysis', 'diagnosing', 'monitoring',
                    'prognosing', 'lung cancer', 'cancer']
        actual = self.tfidf.extract_popular_ngrams(text)

        self.assertGreaterOrEqualDiceScore(expected, actual)

    def test_patent_US06887476_20050503(self):
        text = '''A method and composition for treating a patient suffering from a disease, disorder or condition and 
        associated pain include the administration to the patient of a therapeutically effective amount of a 
        neurotoxin selected from a group consisting of Botulinum toxin types A, B, C, D, E, F and G. '''
        expected = ['treating', 'disease', 'pain', 'administration', 'therapeutically', 'neurotoxin', 'Botulinum toxin']
        actual = self.tfidf.extract_popular_ngrams(text)

        self.assertGreaterOrEqualDiceScore(expected, actual)

    def test_patent_US07862747_20110104(self):
        text = '''An excellent EL device having low driving voltage, high luminous efficiency and long life can be 
        obtained by using a charge-transporting thin film composed of a charge-transporting varnish which contains an 
        arylsulfonic acid compound represented by the formula (1) or (2) below as an electron-acceptor material 
        especially in an OLED device or a PLED device. [In the formulae, X represents O, S or NH; A represents a 
        naphthalene ring or anthracene ring having a substituent other than X and n SO 3 H groups; B represents a 
        substituted or unsubstituted hydrocarbon group, 1,3,5-triazine group or a substituted of unsubstituted group 
        represented by the following formula (3) or (4): 

            (wherein W 1  and W 2  each independently represents O, S, an S(O) group, an S(O 2 ) group, 
            or a substituted or unsubstituted N, Si, P or P(O) group); n indicates the number of sulfonic acid groups 
            bonded to A which is an integer satisfying 1≦n≦4; q indicates the number of B—X bonds which is an integer 
            satisfying 1≦q; and r indicates the number of recurring units which is an integer satisfying 1≦r.]. '''
        expected = ['low driving voltage', 'driving voltage', 'voltage', 'luminous', 'charge-transporting', 'thin film',
                    'film', 'varnish', 'arylsulfonic', 'acid', 'compound', 'acid compound',
                    'electron-acceptor', 'OLED', 'PLED', 'naphthalene ring', 'naphthalene', 'ring', 'anthracene ring',
                    'anthracene', 'substituent', 'hydrocarbon group', 'hydrocarbon', 'triazine group',
                    'triazine', 'sulfonic acid', 'bonds', 'substitude', 'unsubstitute']

        print("test_patent_US07862747_20110104: loads of keywords not-detected here. One to watch!")
        actual = self.tfidf.extract_popular_ngrams(text)

        self.assertGreaterOrEqualDiceScore(expected, actual)

    def test_patent_US08794089_20140805(self):
        text = '''A transmission includes first and second rotating shafts arranged in parallel and multiple gear 
        sets including gears fixed to the first rotating shaft and rotatable gears rotatably supported on the second 
        rotating shaft. Each gear set includes first and second engagement units engaged with first and second 
        engagement-target portions formed on end surfaces of each rotatable gear and transmitting torque in first and 
        second rotational directions; and a shift mechanism axially and independently moving the first and second 
        engagement units. Each one of the first and second engagement units has a slanted portion allowing the 
        engagement unit to withdraw from the rotatable gear when the other one of the engagement units is disengaged 
        and torque transmitted in a corresponding direction is input. Each one of the first and second engagement 
        unit of each gear set moves integrally with the other one of the engagement units of an adjacent gear set. '''
        expected = ['transmission', 'rotating shafts', 'gear sets', 'gear', 'shaft', 'transmitting torque', 'torque',
                    'axially', 'engagement units', 'rotational directions', 'rotatable gear']
        actual = self.tfidf.extract_popular_ngrams(text)

        self.assertGreaterOrEqualDiceScore(expected, actual)

    def test_patent_US07274983_20070925(self):
        text = '''When a sport mode shifting request occurs during gear shifting of an automatic transmission 
        according to a vehicle running state, a transmission control unit determines whether or not to accept the 
        sport mode shifting request on the basis of an expected turbine speed at a target speed of currently 
        controlled upshift and an expected turbine speed at a final target speed requested by the sport mode shifting 
        request. By such a scheme, shift shock is reduced and responsiveness is enhanced. '''
        expected = ['gear shifting', 'automatic transmission', 'transmission', 'transmission control unit',
                    'turbine speed', 'turbine', 'sport mode shifting', 'shift shock']
        actual = self.tfidf.extract_popular_ngrams(text)

        self.assertGreaterOrEqualDiceScore(expected, actual)

    def test_patent_US08415520_20130409(self):
        text = '''The invention is a method for processing a mixture containing water, 3-methyl-1-butane and at least 
        one other methylbutene. The method comprises primary distillation of the mixture, giving a gaseous primary 
        overhead product containing methylbutene and water and a water-free primary bottom product containing 
        3-methyl-1-butene; condensation of the gaseous primary overhead product so as to give a condensate comprising 
        a liquid aqueous phase and a liquid organic phase; separation of the condensate into a liquid aqueous phase 
        and a liquid organic phase; discharge of the liquid aqueous phase; recirculation of the organic phase to the 
        primary distillation; and finally secondary distillation of the water-free primary bottom product from the 
        primary distillation so as to give a secondary overhead product comprising 3-methyl-1-butene and a secondary 
        bottom product. The secondary overhead product obtained has a purity which enables it to be used directly as 
        monomer or comonomer for preparing polymers or copolymers. '''
        expected = ['methylbutene', '3-methyl-1-butane', 'distillation', 'aqueous phase', 'organic phase',
                    'primary distillation', 'secondary distillation', 'overhead product', 'bottom product',
                    'monomer', 'comonomer', 'polymer', 'copolymer']
        actual = self.tfidf.extract_popular_ngrams(text)

        self.assertGreaterOrEqualDiceScore(expected, actual)

    def test_patent_US09140373_20150922(self):
        text = '''A thermally actuated valve is provided herein. The thermally actuated valve includes a valve 
        fitting, a valve body, and a movement control mechanism. The valve fitting includes an active member to 
        selectively activate based on an input. The valve body includes a passive wax member that moves between a 
        passive contraction state and a passive expansion state based on a passive temperature. The movement control 
        mechanism is disposed between the valve fitting and the valve body. The movement control mechanism controls 
        movement of the valve body between an open state and a dosed state based on movement of at least one of the 
        active member and the passive wax member. '''
        expected = ['thermally actuated valve', 'valve fitting', 'passive', 'movement control mechanism',
                    'valve body', 'wax member', 'active member']
        actual = self.tfidf.extract_popular_ngrams(text)

        self.assertGreaterOrEqualDiceScore(expected, actual)

    def test_patent_US09100829_20150804(self):
        text = '''An apparatus and method to control an application in a portable terminal are provided. A method to 
        manage an application includes confirming control information that at least one application installed in the 
        portable terminal requires, selecting the at least one application requiring control information whose use is 
        restricted among a plurality of applications installed in the portable terminal, and restricting a running of 
        the selected at least one application. '''
        expected = ['portable terminal', 'application', 'control information']
        actual = self.tfidf.extract_popular_ngrams(text)

        self.assertGreaterOrEqualDiceScore(expected, actual)

    def test_patent_US08878515_20141104(self):
        text = '''A constant current is provided to an energizing coil in a magnetic detector by charging a capacitor 
        through a resistor from a high voltage source. Discharging of the capacitor into the energizing coil quickly 
        increases current in the energizing coil. After the capacitor is switched off, a low voltage source maintains 
        current constant in the energizing coil. The coil discharges its energy as a negative voltage to the 
        capacitor. A high negative voltage source tops off the capacitor. After a delay, the capacitor discharges a 
        negative current into the energizing coil. A negative low voltage source maintains the negative current. The 
        negative voltage source is disconnected, and the coil discharges positive voltage into the capacitor. The 
        high voltage source tops off the capacitor with positive voltage to repeat the cycle. '''
        expected = ['constant current', 'energizing coil', 'magnetic detector', 'discharge', 'capacitor']
        actual = self.tfidf.extract_popular_ngrams(text)

        self.assertGreaterOrEqualDiceScore(expected, actual)

    def test_patent_US08388340_20130305(self):
        text = '''A method and system are disclosed for detecting an apical position depending on the change in the 
        impedance between a first electrode inserted into the root canal of the tooth of a patient and a second 
        external electrode applied to a body surface of the patient. According to some embodiments, a regulated 
        current such as an alternating current having a substantially constant amplitude is supplied between the two 
        electrodes, and this current serves as a measurement signal. Alternatively or additionally, the frequency of 
        the time varying (e.g. alternating) current is at least 50 KHZ, and/or at most about 300 KHZ. In some 
        embodiments, the presently disclosed device includes a processing unit which determines a 
        capacitance-governed function when the first electrode is in the apical region, and which determined a 
        function at least moderately governed by resistance when the electrode is in the dental neck region. 
        Optionally, the first electrode inserted into the root canal is a dental file or reamer. '''
        expected = ['apical', 'root canal', 'electrode', 'dental neck', 'reamer', 'dental file', 'dental']
        actual = self.tfidf.extract_popular_ngrams(text)

        self.assertGreaterOrEqualDiceScore(expected, actual)

    def test_patent_US09198389_20151201(self):
        text = '''The present invention is in the field of soybean variety WN1116094-1 breeding and development. The 
        present invention particularly relates to the soybean variety WN1116094-1 and its progeny, and methods of 
        making WN1116094-1. '''
        expected = ['soybean', 'soybean variety WN1116094-1', 'WN1116094-1', 'breeding']
        actual = self.tfidf.extract_popular_ngrams(text)

        self.assertGreaterOrEqualDiceScore(expected, actual)

    def test_patent_US09002941_20150407(self):
        text = '''Embodiments of the invention include a method for determining recommended recipients of a 
        communication. The method may include determining one or more attributes for one or more members of a first 
        group. The method may also include receiving a first list of one or more recipients to receive a 
        communication from a sender, wherein the recipients and the sender have a relationship based on the 
        attributes. The method may also include determining a second list of one or more recipients to receive the 
        communication, wherein the recipients of the second list are determined by whether the attributes of the 
        first list recipients, the sender, and the first group members comply with a set of communication rules. '''
        expected = ['recipients', 'recepients communication', 'communication rules', 'recommended recipients',
                    'communication']
        actual = self.tfidf.extract_popular_ngrams(text)

        self.assertGreaterOrEqualDiceScore(expected, actual)

    def test_patent_US08666791_20140304(self):
        text = '''A method, computer program and single enterprise resource planning (ERP) system for procuring a 
        product or service. The single ERP system includes an input for receiving information relating to product 
        demand for a plurality of end users via a cooperative buying group, a predictive engine for accumulating the 
        received product demand into a single demand schedule and an output for transmitting the single demand 
        schedule to a manufacturer/original source of the product. An optimization engine is provided for receiving 
        product availability information from the manufacturer/original source of the product by the cooperative 
        buying group, and for determining one or more options for ordering the products. A procurement system is also 
        included for providing the product availability information to the one or more options for ordering the 
        products, for receiving one or more product orders, for ordering product from the manufacturer/original 
        source of the product, and for managing shipping the product. '''
        expected = ['enterprise resource planning', 'ERP', 'cooperative buying group', 'demand schedule',
                    'product demand', 'product availability', 'manufacturer/original source',
                    'procurement system', 'ordering products', 'shipping product']
        actual = self.tfidf.extract_popular_ngrams(text)

        self.assertGreaterOrEqualDiceScore(expected, actual)

    def test_patent_US09184435_20151110(self):
        text = '''The method for producing an electrode for an electrochemical element of the present invention 
        includes a slurry filling step of filling a slurry containing an active material into continuous pores of an 
        aluminum porous body having the continuous pores, and a slurry drying step of drying the slurry filled, 
        and in this method, after the slurry drying step, an electrode for an electrochemical element is produced 
        without undergoing a compressing step of compressing the aluminum porous body having the slurry filled 
        therein and dried. In the electrode, a mixture containing an active material is filled into continuous pores 
        of an aluminum porous body having the continuous pores, and porosity (%) of the aluminum porous body, 
        the porosity being represented by the following equation, is 15 to 55%. 

            Porosity(%)={1−(volume of electrode material)/(apparent volume of electrode)}×100'''
        expected = ['electrode', 'slurry', 'slurry drying', 'aluminium porous body', 'continuous pores',
                    'electrochemical element', 'porosity', 'porous body']
        actual = self.tfidf.extract_popular_ngrams(text)

        self.assertGreaterOrEqualDiceScore(expected, actual)

    def test_patent_US09589954_20170307(self):
        text = '''Electric-field concentration in the vicinity of a recess is suppressed. A gate insulating film is 
        provided on a substrate that has a drain region and a first recess therein. The first recess is located 
        between the gate insulating film and the drain region, and is filled with an insulating film. The insulating 
        film has a second recess on its side close to the gate insulating film. An angle defined by an inner side 
        face of the first recess and the surface of the substrate is rounded on a side of the drain region close to 
        the gate insulating film. '''
        expected = ['Electric-field concentration', 'gate insulating film', 'recess', 'drain region', 'insulating film']
        actual = self.tfidf.extract_popular_ngrams(text)

        self.assertGreaterOrEqualDiceScore(expected, actual)

    def test_patent_US07788644_20100831(self):
        text = '''A method and system are provided for implementing performance monitoring of an application on a 
        mobile device. An instrumentation tool is provided allowing a user to view the entities in an application 
        file for a mobile device and selecting those entities for which performance monitoring is to be implemented. 
        The instrumentation tool adds performance monitoring methods to the application file and generates a new 
        instrumented application file that is transferred to the mobile device. When the instrumented application 
        file is executed on the mobile device, the performance monitoring methods instrumented into the file execute 
        generating data in a performance log file that is stored on the mobile device. This performance log file may 
        be transferred to a remote device for further analysis in addition to viewing the performance log file on the 
        mobile device. The user selected entities for performance monitoring in the application file may be saved to 
        a configuration file that can later be loaded and modified by the user to facilitate further performance 
        monitoring of an application. '''
        expected = ['performance monitoring', 'mobile device', 'instrumentation tool', 'application file',
                    'instrumented application file', 'performance log', 'configuration file']
        actual = self.tfidf.extract_popular_ngrams(text)

        self.assertGreaterOrEqualDiceScore(expected, actual)

    def test_patent_US07765845_20100803(self):
        text = '''The stem slide device of the present invention in an extruder has a stem vertical movement support 
        ( 71 ) and slide guide members ( 72 - 1, 2 ) fastened to the support and forming guide grooves. A stem ( 6 ) 
        pushing against a billet loaded in a container is held at the slide table ( 73 ) horizontally. The slide 
        table slides in the vertical direction along the vertical face of the support. When the slide table is 
        positioned at the bottom end of the guide groove, the hydraulic cylinders ( 77 - 1, 2 ) are driven so that 
        the rods ( 78 - 1, 2 ) push the back surface of the slide table against the vertical face of the support. 
        Therefore, the axis of the stem is held matched with the axis of the container. '''
        expected = ['stem slide device', 'extruder', 'billet', 'slide table', 'hydraulic cylinders']
        actual = self.tfidf.extract_popular_ngrams(text)

        self.assertGreaterOrEqualDiceScore(expected, actual)

    def test_patent_US09767335_20170919(self):
        text = '''For reading optical information arranged on a target object by analyzing an image of the target 
        object captured through an imaging device, a plurality of blocks are disposed in an area of the image in 
        which the optical information is arranged such that the plurality of blocks cover a whole range in the 
        arrangement direction of the optical information, each of the blocks being in a parallelogram shape in which 
        facing two sides are in parallel with a pixel arrangement direction of the image and the other two sides are 
        vertical to the arrangement direction of the optical information, a first arrangement data indicating 
        arrangement of the optical information in the block is generated for each of the blocks based on the image 
        data of the block, and a second arrangement data indicating arrangement of the whole optical information is 
        generated by combining the generated respective first arrangement data. '''
        expected = ['optical information', 'imaging device', 'pixel arrangement direction']
        actual = self.tfidf.extract_popular_ngrams(text)

        self.assertGreaterOrEqualDiceScore(expected, actual)

    def test_patent_US08683258_20140325(self):
        text = '''A method for fast I/O path failure detection and cluster wide failover. The method includes 
        accessing a distributed computer system having a cluster including a plurality of nodes, and experiencing an 
        I/O path failure for a storage device. An I/O failure message is generated in response to the I/O path 
        failure. A cluster wide I/O failure message broadcast to the plurality of nodes that designates a faulted 
        controller. Upon receiving I/O failure responses from the plurality of nodes, an I/O queue message is 
        broadcast to the nodes to cause the nodes to queue I/O through the faulted controller and switch to an 
        alternate controller. Upon receiving I/O queue responses from the plurality of nodes, an I/O failover commit 
        message is broadcast to the nodes to cause the nodes to commit to a failover and un-queue their I/O. '''
        expected = ['I/O', 'I/O path failure detection', 'distributed computer system', 'cluster', 'I/O failure',
                    'message broadcast', 'failover', 'plurality of nodes']
        actual = self.tfidf.extract_popular_ngrams(text)

        self.assertGreaterOrEqualDiceScore(expected, actual)

    def test_patent_US07341610_20080311(self):
        text = '''A fuel cell system includes a fuel cell unit and a gas-generating system containing at least one 
        reforming unit for obtaining a hydrogen-rich reformate from a fuel. It is possible to supply the reformate at 
        least partly to the anode side of the fuel cell unit. The system may include a first reforming reactor for 
        producing a first reformate with a high outlet temperature; a second reforming reactor for producing a second 
        reformate with a second outlet temperature which is below the first outlet temperature; a mixing element for 
        mixing the first reformate with at least one fuel and located between an outlet of the first reforming 
        reactor and an inlet of the second reforming reactor. The second reformate may be supplied to a 
        gas-purification system and the purified reformate supplied to the fuel cell unit. '''
        expected = ['fuel cell', 'gas-generating system', 'reforming unit', 'reforming', 'hydrogen-rich reformate',
                    'reformate', 'reforming reactor', 'mixing element', 'gas-purification system']
        actual = self.tfidf.extract_popular_ngrams(text)

        self.assertGreaterOrEqualDiceScore(expected, actual)

    def test_patent_US09867990_20180116(self):
        text = '''Aspects of the present disclosure include a medical device system including an implantable medical 
        device and an external device with three or more electrodes configured to contact a patient's skin. The 
        external device either transmits or receives a test signal to or from the implantable medical device using a 
        plurality of possible receive dipoles, where each possible receive dipole is formed by a pair of electrodes. 
        A signal quality monitor, either at the implantable medical device or at the external device, measures a 
        signal quality for the possible receive dipoles. '''
        expected = ['implantable medical device', 'medical device', 'receive dipoles', 'pair of electrodes',
                    'signal quality']
        actual = self.tfidf.extract_popular_ngrams(text)

        self.assertGreaterOrEqualDiceScore(expected, actual)

    def test_patent_US09351863_20160531(self):
        text = '''A system for inducing weight loss ina patient includes (a) an elongated element extending from a 
        proximal end to a distal end, the elongated element separating a duodenum into first and second channels, 
        wherein the elongated element is expandable; and (b) a retainer connected to the elongated element and 
        securing the elongated element in a desired orientation within the duodenum. In the desired orientation, 
        all chyme flowing through the duodenum enters the first channel and the second channel is open to a papilla 
        of vater so that digestive fluids from the papilla of vater enter the second channel. '''
        expected = ['inducing weight loss', 'duodenum', 'chyme', 'digestive fluids', 'papilla of vater',
                    'elongated element']
        actual = self.tfidf.extract_popular_ngrams(text)

        self.assertGreaterOrEqualDiceScore(expected, actual)

    def test_patent_US07825904_20101102(self):
        text = '''An information processing apparatus includes a touch panel provided on a menu selection screen 
        including a plurality of items, a direction key for instructing a moving direction of a cursor, 
        and an execution key for instructing execution of a process corresponding to a selecting item. The display of 
        the cursor can be moved according to a position instructed by the touch panel as well as the direction key. A 
        process corresponding to a selected item is executed by a touch-off as well as by an operation of the 
        execution key. It is noted that in a case that a touch input is continued for equal to or more than a 
        predetermined time period after the touch-on, in a case that a touch input is present until the touch-off in 
        an area except for the area corresponding to the item pointed at a start of touch, or in a case that a touch 
        input is present until the touch-off in an area at a predetermined distance away from the position instructed 
        at a start of touch, the process corresponding to the selected item is not executed. '''
        expected = ['information processing', 'touch panel', 'menu selection', 'direction key', 'execution key',
                    'touch', 'execution', 'selected item']
        actual = self.tfidf.extract_popular_ngrams(text)

        self.assertGreaterOrEqualDiceScore(expected, actual)

    def test_patent_US08985196_20150324(self):
        text = '''A heat dissipation device with mounting structure includes a main body and a plurality of mounting 
        elements. The main body includes an internally defined chamber having a first side and an opposite second 
        side; a plurality of supports located in the chamber and respectively having two ends connected to the first 
        side and the second side of the chamber; a working fluid filled in the chamber; and a plurality of connection 
        sections in the form of recesses formed on an outer surface of the main body at positions corresponding to 
        the supports in the chamber. The mounting elements are connected to the connection sections. With these 
        arrangements, the heat dissipation device with the mounting elements connected to one outer surface thereof 
        can maintain the chamber in the main body in an airtight state and ensure tight contact of it with a 
        heat-generating element. '''
        expected = ['heat', 'mounting', 'chamber', 'fluid', 'airtight', 'heat dissipation', 'airtight state',
                    'heat generating']
        actual = self.tfidf.extract_popular_ngrams(text)

        self.assertGreaterOrEqualDiceScore(expected, actual)

    def test_patent_US07372372_20080513(self):
        text = '''An automatic meter reading system that includes a head end controller and an endpoint that is 
        interfaced to a utility meter. The head end controller and the endpoint communicate via RF communication. The 
        endpoint includes an internal clock that synchronizes itself to a clock countdown signal. The clock countdown 
        signal is generated by the head end controller through use of sequence inversion keying. '''
        expected = ['meter reading system', 'meter', 'head end controller', 'utility meter', 'RF communication',
                    'clock', 'clock countdown signal', 'clock countdown', 'sequence inversion keying',
                    'sequence inversion', 'inversion', 'utility', 'signal']
        actual = self.tfidf.extract_popular_ngrams(text)

        self.assertGreaterOrEqualDiceScore(expected, actual)

    def test_patent_US08495473_20130723(self):
        text = '''A transceiver is designed to share memory and processing power amongst a plurality of transmitter 
        and/or receiver latency paths, in a communications transceiver that carries or supports multiple 
        applications. For example, the transmitter and/or receiver latency paths of the transceiver can share an 
        interleaver/deinterleaver memory. This allocation can be done based on the data rate, latency, BER, 
        impulse noise protection requirements of the application, data or information being transported over each 
        latency path, or in general any parameter associated with the communications system. '''
        expected = ['transceiver', 'memory', 'processing power', 'transmitter', 'receiver', 'latency paths', 'latency',
                    'interleaver/deinterleaver', 'BER', 'impulse', 'impulse noise protection', 'impulse',
                    'communications system']
        actual = self.tfidf.extract_popular_ngrams(text)

        self.assertGreaterOrEqualDiceScore(expected, actual)

    def test_patent_US09740641_20170822(self):
        text = '''An information processing device that are capable of continuing access to an I/O device by 
        operational computers even when a failure has occurred in a management computer is provided. A virtualization 
        app_bridge ( 300 ) includes a monitoring unit ( 307 ) and a backup control unit ( 308 ). The virtualization 
        app_bridge ( 300 ) provides operational computers ( 200 ) with virtual functions of an I/O device ( 400 ). The 
        monitoring unit ( 307 ) detects failures in a management computer ( 100 ). The backup control unit ( 308 ) 
        generates backup management information ( 341 ) on the basis of packets transmitted and received between the 
        management computer ( 100 ) and the I/O device ( 400 ), and, when a failure in the management computer ( 100 
        ) is detected by the monitoring unit ( 307 ), controls the I/O device  400  on the basis of the backup 
        management information ( 341 ) in place of the management computer ( 100 ). '''
        expected = ['I/O', 'management', 'virtualization app_bridge', 'monitoring', 'backup', 'control', 'transmitted',
                    'packets', 'I/O device', 'virtual', 'virtual functions', 'backup management information']
        actual = self.tfidf.extract_popular_ngrams(text)

        self.assertGreaterOrEqualDiceScore(expected, actual)

    def test_patent_US09063872_20150623(self):
        text = '''A method of performing forward error correction with configurable latency, where a configurable 
        latency algorithm evaluates a target Bit Error Rate (BER) against an actual BER and adjusts the size of a 
        configurable buffer such that the target BER may be achieved when utilizing the smallest buffer size 
        possible. When errors are corrected without the utilization of each of the configurable buffer locations, 
        the algorithm reduces the size of the buffer by y buffer locations; the algorithm may continue to 
        successively reduce the size of said buffer until the minimum number of buffer locations are utilized to 
        achieve the target BER. If the buffer locations have been reduced such that the buffer size is too small and 
        the target BER cannot be achieved, the algorithm may increase the size of the buffer until the minimum number 
        of buffer locations are utilized to achieve the target BER. '''
        expected = ['forward error', 'error', 'configurable latency', 'configurable', 'Bit Error Rate', 'latency',
                    'algorithm', 'utilization', 'configurable buffer locations', 'buffer', 'configurable buffer',
                    'locations', 'BER', 'target']
        actual = self.tfidf.extract_popular_ngrams(text)

        self.assertGreaterOrEqualDiceScore(expected, actual)

    def test_patent_US08793346_20140729(self):
        text = '''A computer-implemented process, computer program product, and apparatus for identifying session 
        identification information. A recording is initiated and an operation sequence of interest is performed while 
        recording and the recording ceases. Responsive to a determination that the operation sequence of interest was 
        successful, information from the operation sequence of interest is saved as recorded information and 
        responsive to a determination that a same operation sequence of interest was recorded, the recorded 
        information from each operation sequence of interest is compared. Differences in the recorded information are 
        identified to form identified differences and a session identifier is constructed using the identified 
        differences. '''
        expected = ['computer-implemented', 'process', 'computer program', 'apparatus', 'identifying', 'session',
                    'recording', 'operation', 'sequence', 'identification information', 'Responsive', 'determination',
                    'operation sequence', 'identfier', 'identified differences']
        actual = self.tfidf.extract_popular_ngrams(text)

        self.assertGreaterOrEqualDiceScore(expected, actual)

    def test_patent_US09430884_20160830(self):
        text = '''A device includes a processor in communication with a first port, a second port and a third port. 
        The first port is configured to communicate via a diagnostic cable with an electronic control unit of a 
        vehicle. The diagnostic able cable has a first end configured to connect to a vehicle port and a second end 
        configured to connect to the first port. The processor is configured to selectively test the communication 
        capability of the diagnostic cable when the first end of the diagnostic cable is connected to the third port 
        and the second end of the diagnostic cable is connected to the first port. '''
        expected = ['processor', 'communication', 'port', 'diagnostic cable', 'diagnostic', 'electronic control',
                    'electronic', 'vehicle', 'configured', 'selectively']
        actual = self.tfidf.extract_popular_ngrams(text)

        self.assertGreaterOrEqualDiceScore(expected, actual)

    def test_patent_US07949833_20110524(self):
        text = '''A digital system that connects to a bus that employs physical addresses comprises a processing 
        core. A level one (L1) cache communicates with the processing core. A level two (L2) cache communicates with 
        the L1 cache. Both the L1 cache and the L2 cache are indexed by virtual addresses and tagged with virtual 
        addresses. A bus unit communicates with the L2 cache and with the bus. '''
        expected = ['digital', 'bus', 'employs', 'physical', 'core', 'processing core', 'L1', 'cache', 'L2', 'indexed',
                    'virtual addresses', 'virtual', 'tagged', 'addresses']
        actual = self.tfidf.extract_popular_ngrams(text)

        self.assertGreaterOrEqualDiceScore(expected, actual)

    def test_patent_US07032335_20060425(self):
        text = '''An apparatus for connecting an implement to a prime mover, the connection apparatus including a 
        body arranged to be mounted on the prime mover. The body includes a connection device for connecting the body 
        to the implement; the connection device comprising a locking member adapted to move to a first position, 
        in which the locking member engages the implement to lock the implement and the body together, 
        and said locking member also being adapted to move to a second position in which the locking member is 
        disengaged from the implement so that the implement can be demounted from the body. '''
        expected = ['apparatus', 'prime mover', 'prime', 'body', 'mounted', 'implement', 'connection', 'locking member',
                    'locking', 'adapted', 'disengaged', 'demounted']
        actual = self.tfidf.extract_popular_ngrams(text)

        self.assertGreaterOrEqualDiceScore(expected, actual)

    def test_patent_US08918582_20141223(self):
        text = '''A virtual EEPROM driver is simulated for a virtual switch. A write function may be written to a 
        shared memory device and designated as a virtual EEPROM driver. The virtual EEPROM driver may be duplicated 
        into a non-volatile memory providing availability during a boot process. '''
        expected = ['virtual', 'virtual EEPROM driver', 'EEPROM', 'driver', 'virtual switch', 'simulated', 'memory',
                    'memory device', 'duplicated', 'non-volatile', 'boot process']
        actual = self.tfidf.extract_popular_ngrams(text)

        self.assertGreaterOrEqualDiceScore(expected, actual)

    def test_patent_US09504699_20161129(self):
        text = '''The present invention refers to the treatment of a rheumatic disease and/or osteoarthritis by 
        administering a delayed-release dosage form of a glucocorticoid to a subject in need thereof. '''
        expected = ['treatment', 'rheumatic disease', 'rheumatic', 'osteoarthritis', 'delayed-release', 'dosage',
                    'glucocorticoid', 'subject']
        actual = self.tfidf.extract_popular_ngrams(text)

        self.assertGreaterOrEqualDiceScore(expected, actual)

    def test_patent_US09056213_20150616(self):
        text = '''Protective barriers are commonly installed beneath ceilings when construction work is performed 
        either on these ceilings or on the roofs located above them. These protective barriers can be comprised 
        entirely of one material or of different materials connected by seams. Some or all of these materials can be 
        designed to fail when subjected to temperatures above a certain temperature range causing melting or some 
        other destructive process to occur to these materials. These failures can create access points from the 
        ceiling through the protective barrier to areas below being protected by the barrier, which can allow water 
        from a fire suppression system, typically located near the ceiling, to reach a fire located below the 
        protective barrier. '''
        expected = ['Protective', 'barriers', 'ceilings', 'construction', 'roofs', 'protective barriers', 'material',
                    'seams', 'fail', 'temperatures', 'melting', 'destructive', 'access points', 'water', 'fire',
                    'fire suppression']
        actual = self.tfidf.extract_popular_ngrams(text)

        self.assertGreaterOrEqualDiceScore(expected, actual)

    def test_patent_US08356946_20130122(self):
        text = '''The interchangeable wheel bearing unit has a wheel hub and two tapered roller bearings, each having 
        an outer ring and an inner ring, between which a respective row of tapered rollers is situated, 
        and a securing ring in at least one of the inner rings of the tapered roller bearing. To assemble the wheel 
        bearing unit without special tools, each outer ring of the tapered roller bearings has a cylindrical 
        extension, which runs coaxially with the wheel hub axis towards the outer face of the bearing and into which 
        a respective seal is inserted, and a retaining element, which is supported on the corresponding inner ring 
        and axially fixes the outer ring, is located on the opposite face of the respective tapered roller bearing 
        from the seal. '''
        expected = ['interchangeable', 'bearing', 'wheel', 'hub', 'tapered roller bearings', 'tapered', 'roller',
                    'outer ring', 'inner ring', 'ring', 'respective', 'assemble', 'tools', 'cylindrical extension',
                    'cylindrical', 'coaxially', 'retaining', 'retaining element', 'axially fixes']
        actual = self.tfidf.extract_popular_ngrams(text)

        self.assertGreaterOrEqualDiceScore(expected, actual)

    def test_patent_US07349344_20080325(self):
        text = '''An access network includes a test system controller that provides a test request in Signaling 
        Network Management Protocol (SNMP) messages to an element management system. A network gateway, 
        in conjunction with the element management system, provides test commands to a customer gateway over a Local 
        loop Emulation Service Embedded Operations Channel (LES-EOC). The customer gateway performs a subscriber loop 
        test on derived subscriber lines connected therewith. Results of the subscriber loop test are provided over 
        the LES-EOC to the gateway. The network gateway sends the results to the test system controller through the 
        element management system in SNMP messages. '''
        expected = ['controller', 'test', 'Signaling', 'SNMP', 'Network', 'customer gateway', 'customer', 'Local loop',
                    'Emulation', 'LES-EOC', 'subscriber', 'gateway', 'network', 'Service']
        actual = self.tfidf.extract_popular_ngrams(text)

        self.assertGreaterOrEqualDiceScore(expected, actual)

    def test_patent_US07476472_20090113(self):
        text = '''The present invention provides a method for designing a mask. First, a main pattern including at 
        least a strip pattern is formed on the mask substrate. A shift feature is added to one end of the strip 
        pattern of the main pattern. Either the phase shift or the optical transmission or both of the shift feature 
        can be adjusted to optimize the resultant critical dimension between line-ends of the main pattern, 
        thus improving pullback of the line-ends of the strip pattern in the main pattern. '''
        expected = ['mask', 'pattern', 'substrate', 'strip', 'strip feature', 'optical transmission', 'optical',
                    'transmission', 'optimize', 'critical', 'dimension', 'line-ends']
        actual = self.tfidf.extract_popular_ngrams(text)

        self.assertGreaterOrEqualDiceScore(expected, actual)

    def test_patent_US09503700_20161122(self):
        text = '''A phosphor wheel includes a rotating disk and a wavelength converting layer. The rotating disk has 
        a first surface and a second surface opposite to the first surface, in which the first surface forms a 
        coating region and a non-coating region. The wavelength converting layer is formed on the coating region of 
        the first surface for converting a light wavelength of a light beam. In addition, an embodiment of the 
        invention discloses a projection device having the phosphor wheel. When the rotating disk of the phosphor 
        wheel rotates, the recess portion may disturb the air around the phosphor wheel such that the temperature of 
        the wavelength converting layer may be effectively decreased. Simultaneously, the rotating disk has a stable 
        dynamic balance and the rotating disk has a larger heat dissipating region because the recess portion is 
        disposed on the rotating disk. '''
        expected = ['phospher', 'wheel', 'rotating disk', 'wavelength', 'rotating', 'coating', 'non-coating',
                    'converting', 'light beam', 'projection device', 'phospher wheel', 'recess', 'temperature',
                    'dynamic balance', 'dynamic', 'heat dissipating', 'heat']
        actual = self.tfidf.extract_popular_ngrams(text)

        self.assertGreaterOrEqualDiceScore(expected, actual)

    def test_patent_US07253825_20070807(self):
        text = '''A method and an apparatus for driving an image display device, such as, a plasma display panel, 
        to represent a gradation. The pixels of a panel are classified into a plurality of groups, and one frame 
        period is divided with time into n subfields. An address period and a sustain period are sequentially 
        executed on each of the groups during at least two of the n subfields. While the pixels of one group are 
        undergoing an address period during a subfield, the pixels of the other groups remain idle. While the pixels 
        of one group are undergoing a sustain period, the pixels of groups that have already been addressed also 
        undergo the sustain period. During one subfield, different gradation weights are allocated to the groups. A 
        gradation of visual brightness for each pixel is determined by performing an address period for either all or 
        some of the groups during at least two subfields. The panel driving method for representing gradation is 
        adaptable. '''
        expected = ['apparatus', 'driving', 'plasma display panel', 'plasma', 'gradation', 'pixels', 'subfields',
                    'sequentially', 'idle', 'sustain period', 'gradation weights', 'visual', 'adaptable']
        actual = self.tfidf.extract_popular_ngrams(text)

        self.assertGreaterOrEqualDiceScore(expected, actual)

    def test_patent_US09435670_20160906(self):
        text = '''A portable apparatus for sensing position of an oil port on a rotating element (e.g. the wheel or 
        track of a heavy vehicle) has a position sensor in electrical communication with a wireless transmitter. The 
        wireless transmitter is configured to receive signals from the position sensor and to send signals to a 
        wireless receiver. The apparatus further has a mounting structure (e.g. a magnet) for temporarily mounting 
        the apparatus on the rotating element. An indicium on the apparatus associated with a desired service action 
        to be performed on the rotating element correlates an angular position of the oil port to a pre-determined 
        reference position of the position sensor, the angular position of the oil port being a correct position for 
        performing the desired service action. In use, the apparatus is mounted on the rotating element so that the 
        indicium points at the oil port and then the rotating element is rotated until the receiver indicates that 
        the oil port is in the correct position. '''
        expected = ['portable', 'apparatus', 'oil', 'rotating', 'wheel', 'vehicle', 'sensor', 'transmitter', 'wireless',
                    'signals', 'receiver', 'mounting', 'magnet', 'rotating element', 'indicium', 'correlates',
                    'angular', 'reference position', 'oil port']
        actual = self.tfidf.extract_popular_ngrams(text)

        self.assertGreaterOrEqualDiceScore(expected, actual)

    def test_patent_US06959911_20051101(self):
        text = '''A valve features a deflectable actuating element ( 1 ) that controls the movements of at least one 
        actual sealing element ( 3 ) for opening and/or closing at least one sealing contour ( 6 ), and the actuating 
        element is loaded via an elastic element ( 5 ) essentially perpendicularly relative to the direction of the 
        deflection; and the longitudinal axis of the actuating element and the force applied by the elastic element 
        are aligned in a position of the actuating element that is between its two extreme positions. The actuating 
        element ( 1 ) is preferably an unilaterally loaded piezoelectric bending transducer, and the sealing element 
        ( 3 ) is preferably a toggle that has the ability to swing around an axis arranged perpendicularly relative 
        to the direction of the deflection of the actuating element ( 1 ). '''
        expected = ['valve', 'deflectable', 'actuating element', 'sealing', 'contour', 'elastic', 'deflection', 'force',
                    'piezoelectric', 'transducer', 'perpendicularly']
        actual = self.tfidf.extract_popular_ngrams(text)

        self.assertGreaterOrEqualDiceScore(expected, actual)

    def test_patent_US09289910_20160322(self):
        text = '''Two surfaces forming a cutting edge and a ridge of a cutting edge existing along the boundary 
        between the two surfaces intersecting with each other are irradiated with a gas cluster ion beam at the same 
        time, the maximum height of the profile of the two surfaces being equal to or smaller than 1 μm. A facet is 
        newly formed on the ridge of the cutting edge by performing the irradiation with the gas cluster ion beam in 
        such a manner that the two surfaces are not perpendicularly but obliquely irradiated with the gas cluster ion 
        beam, and at least a part of the ridge of the cutting edge is perpendicularly irradiated with the gas cluster 
        ion beam. '''
        expected = ['ridge', 'cutting edge', 'intersecting', 'gas', 'gas cluster', 'ion', 'ion beam', 'irradiation',
                    'perpendicularly']
        actual = self.tfidf.extract_popular_ngrams(text)

        self.assertGreaterOrEqualDiceScore(expected, actual)

    def test_patent_US07682616_20100323(self):
        text = '''Phytoceutical compositions for the prevention and treatment of circulatory disorders, feminine 
        endocrine disorders, and dermal disorders. A specific combination of extracts of plants is taught, 
        as well as principles for varying the formulations based on categorizing plants into one of three groups, 
        Energy, Bio-Intelligence, and Organization and selecting several plants from each group. Such combinations 
        have synergistic effects, with minimal side effects. '''
        expected = ['Phytoceutical', 'treatment', 'circulatory disorders', 'feminine', 'endocrine', 'disorders',
                    'dermal disorders', 'extracts', 'plants', 'forumulations', 'categorizing', 'Energy',
                    'Bio-Intelligence', 'synergistic', 'side effects']
        actual = self.tfidf.extract_popular_ngrams(text)

        self.assertGreaterOrEqualDiceScore(expected, actual)

    def test_patent_US07872489_20110118(self):
        text = '''A method of locating a defect of a failed semiconductor device which includes applying a test 
        pattern to the failed semiconductor device and providing failed semiconductor device test responses as a pass 
        signature, applying radiation to each of multiple locations of circuitry of a correlation semiconductor 
        device with sufficient energy to induce a fault in the circuitry, applying the test pattern to the 
        correlation semiconductor device while the radiation is applied to the location and comparing correlation 
        semiconductor device test responses with the pass signature for each location, and determining a defect 
        location of the failed semiconductor device in which correlation semiconductor device test responses at least 
        nearly match the pass signature. The radiation may be a laser beam. The method may include determining an 
        exact match or a near match based on a high correlation result. Asynchronous scanning may be used to provide 
        timing information. '''
        expected = ['defect', 'semiconductor', 'semiconductor device', 'signature', 'radiation', 'circuitry',
                    'correlation', 'laser beam', 'laser', 'Asynchronous']
        actual = self.tfidf.extract_popular_ngrams(text)

        self.assertGreaterOrEqualDiceScore(expected, actual)

    def test_patent_US06932708_20050823(self):
        text = '''This communication game system comprises a client system  1  and a game server system  2  
        for communicating with the client system  1 . The game server system  2  comprises a database  21  for storing group 
        information which relates a plurality of client systems to each other as a battle group. The game server system  2  
        is structured to decide a battle combination among the client systems  1  belonging to the same battle group, to perform a 
        battle by managing the sending and receiving of data between the client systems determined by the above-mentioned combination, 
        and to decide the next combination in accordance with the results of the battle. Each client system  1  has its own character select function and chat function when watching games.'''
        expected = ['client', 'system', 'server', 'database', 'battle', 'battle group', 'chat', 'game', 'function',
                    'client system', 'server system', 'battle combination', 'chat function']
        actual = self.tfidf.extract_popular_ngrams(text)

        self.assertGreaterOrEqualDiceScore(expected, actual)

        # FN database

    def test_patent_US08306135_20121106(self):
        text = '''A communication system using an OFDM includes a data creation section for coding data to be transmitted and mapping 
        the data, a null symbol insertion section for filling a null symbol into a no-data subchannel if the number of subchannels containing 
        the mapped data is small for the band assignment, and a symbol interleave section for performing symbol interleave in the whole user assignment band and 
        inserting a known training symbol and pilot symbol into the determined symbol position of the user assignment band are included and symbols are placed such that 
        signal phase change is continuous in the same subcarrier between symbols and carrier sense is executed at the positions of the symbols where the signal phase change is continuous.'''
        expected = ['communication', 'OFDM', 'data', 'coding', 'transmit', 'mapp', 'no data', 'subchannel', 'band',
                    'interleav', 'assign', 'insert', 'signal', 'phase', 'subcarrier', 'carrier', 'continuous',
                    'communication system', 'null symbol', 'band assign', 'phase change', 'training symbol',
                    'pilot symbol', 'map', 'map data']
        actual = self.tfidf.extract_popular_ngrams(text)

        self.assertGreaterOrEqualDiceScore(expected, actual)

    def test_patent_US09395838_20160719(self):
        text = '''An input device is equipped with a touch panel for detecting an input, a coordinates acquiring unit for detecting input coordinates which are coordinates 
        of the input detected by the touch panel, and a pull manipulation judging unit which, when an input to an input detection surface which is a surface on which the touch panel 
        is placed, makes effective the Z coordinate in the direction perpendicular to the input detection surface among the input coordinates detected by the coordinates acquiring unit.'''
        expected = ['input', 'touch', 'panel', 'touch panel', 'coordinate', 'Z']
        actual = self.tfidf.extract_popular_ngrams(text)

        self.assertGreaterOrEqualDiceScore(expected, actual)

        # FN: Z

    def test_patent_US09714145_20170725(self):
        text = '''The disclosure describes, in part, a system and method for improving the stacking of containers on or in a transportation unit. 
        In some implementations, a stacking configuration may be planned that identifies containers and a position for those containers in the stacking configuration. 
        The stacking configuration may be planned based on dimension values of the containers such that when stacked the stacking configuration remains stable. 
        In addition, to improve the efficiency at which containers may be stacked, the disclosure describes that containers and/or the picking of items for those containers may be 
        sequenced so that the containers, when packed and routed, arrive in a manner that allows efficient stacking.'''
        expected = ['stack', 'container', 'configur', 'sequenc', 'transportation unit', 'transportation']
        actual = self.tfidf.extract_popular_ngrams(text)

        self.assertGreaterOrEqualDiceScore(expected, actual)

        # FP: {'configur', 'disclosur', 'describ', 'improv', 'effici'} (too general)
        # TN: plan
        # Thoughts: need to consider synonyms

    def test_patent_US09910243_20180306(self):
        text = '''A lens interchangeable type camera system, comprising an interchangeable lens and a camera body, comprising a first control section that carries out manual focus 
        control by detecting rotation direction and rotation amount of an operation member, in accordance with a manual focus mode command from a mode setting at a time when the 
        operation member is at the first position, and a second control section that, when the operation member is at a second position, irrespective of a command from a mode setting section, 
        notifies a detection result of a first detection section to the camera body, detects rotational position of the operation member using a third detection section, and forcibly carries 
        out manual focus control based on a rotation position, wherein the lens interchangeable type camera system further comprises a function restriction section that sets operation of the 
        second control section to valid or invalid.'''
        expected = ['lens', 'camera', 'interchangeable', 'focus', 'rotation', 'manual focus']
        actual = self.tfidf.extract_popular_ngrams(text)

        self.assertGreaterOrEqualDiceScore(expected, actual)

    def test_patent_US07198133_20070403(self):
        text = '''A one-piece, transparent flexible ear coupler for use with hearing evaluation is disclosed. It includes an annular side wall and a bottom wall forming an acoustic chamber.
         A flexible adhesive-backed flange is disposed on the periphery of the ear coupler. The flange attaches to the subject's head, firmly holding the ear coupler in place over the ear. 
         The annular side wall has a port for the placement of a transducer assembly, and also has ribs to help lock the transducer assembly in place. The transducer assembly can be placed in 
         an up or down position, and can be switched between positions while the coupler is attached to the subject's head. The ear coupler advantageously conforms to the subject's head, thereby minimizing the likelihood that the ear coupler will become detached during testing. The coupler can be inexpensively manufactured, since its one-piece design allows the use of relatively low-cost processes such as injection molding and thermoforming.'''
        expected = ['ear', 'coupler', 'hear', 'evaluat', 'transducer', 'head', 'acoustic chamber', 'flange',
                    'transducer assembly', 'ear coupler']
        actual = self.tfidf.extract_popular_ngrams(text)

        self.assertGreaterOrEqualDiceScore(expected, actual)

        # Code fails

    def test_patent_US09039285_20150526(self):
        text = '''A synthetic resin-made thrust sliding bearing  1  includes a synthetic resin-made upper casing  2  which is fixed to a vehicle body side via a mounting member; 
        a synthetic resin-made lower casing  3  which is superposed on the upper casing  2  so as to be rotatable about an axis O in a circumferential direction R relative to the upper casing
        2 ; and a synthetic resin-made sliding bearing piece  5  disposed in a space  4  between the upper casing  2  and the lower casing  3.'''
        expected = ['synthetic', 'resin', 'thrust', 'slid', 'bearing', 'vehicle body', 'mounting member',
                    'circumferential direction']
        actual = self.tfidf.extract_popular_ngrams(text)

        self.assertGreaterOrEqualDiceScore(expected, actual)

        # FP: {'slide', '3', 'direct', 'lower', 'upper', 'dispo', '2', 'space', 'case', 'circumferenti'} (too general, positional terms)

    def test_patent_US07603640_20091013(self):
        text = '''To generate a floorplan for an integrated circuit to be formed by a collection of modules interconnected by nets, 
        the floorspace to be occupied by the integrated circuit is partitioned into regions and all of the modules are allocated among those regions. 
        The regions are then iteratively partitioning into smaller progressively smaller regions with modules previously allocated any partitioned region allocated among the regions 
        into which it was partitioned, until each region of the floorplan has been allocated no more than a predetermined maximum number of modules. A separate floorplan is then generated 
        for each region. Neighboring regions are then iteratively merged to create progressively larger regions, until only a single region remains, wherein upon merging any neighboring regions 
        to form a larger merged region, the floorplans of the neighboring regions are merged and refined to create a floorplan for the merged region.'''
        expected = ['circuit', 'floorplan', 'module', 'integrated circuit']
        actual = self.tfidf.extract_popular_ngrams(text)

        self.assertGreaterOrEqualDiceScore(expected, actual)

        # FP: {'iter', 'larger', 'neighbor', 'region', 'merg', 'alloc', 'progress', 'partit'}  (too general, positional terms)

    def test_patent_US08389238_20130305(self):
        text = '''Efficient and prolonged hCFTR expression is one of the major obstacles for cystic fibrosis lung therapy. hCFTR mRNA expression levels depend on eukaryotic expression 
        cassette components, prokaryotic backbone elements, and the gene transfer method may also influence transcriptional silencing mechanisms. A codon-optimized and CpG-reduced human CFTR 
        gene (CO-CFTR) was made. Various vector modifications were tested to facilitate extended duration of CO-CFTR expression. Insertion of an extended 3′BGH transcribed sequence (712 bp) 
        in an inverted orientation produced prolonged expression of CO-CFTR expression at biologically relevant levels. Further studies revealed that prolonged CO-CFTR expression is dependant on 
        the orientation of the extended BGH 3′ BGH transcribed sequence and its transcription, is not specific to the UbC promoter, and is less dependent on other vector backbone elements.'''

        expected = ['hCFTR', 'expression', 'cystic fibrosis', 'lung', 'therapy', 'lung therapy', 'mRNA', 'eukaryotic',
                    'cassette', 'prokaryotic', 'backbone', 'gene', 'transfer', 'transcript', 'codon', 'CpG', 'CTFR',
                    'CO-CTFR', 'vector', "BGH", 'UbC', 'promoter', 'sequence', 'modification']
        actual = self.tfidf.extract_popular_ngrams(text)

        self.assertGreaterOrEqualDiceScore(expected, actual)

        # FN: {'therapi', 'lung', 'modif', 'ubc', 'backbon', 'promot', '3', 'cassett', 'prokaryot', 'bp', '712', 'fibrosi', 'cpg', 'bgh', 'codon', 'transcript', 'transfer', 'cystic', 'mrna', 'eukaryot', 'co-ctfr', 'hcftr', 'ctfr', "'"}

        # In this case, incorporating a lot of biotech technical words, I selected many such words not picked up by the computer

    def test_patent_US07377274_20080527(self):
        text = '''An automatic, rapid-firing toy gun is powered by a fast moving air stream. The toy gun is simple in design and does not require a lot of effort and time to fire the projectiles 
        or to load the projectiles between firing. The toy gun includes a barrel, a fan, a loading chamber, and a trigger. The barrel has a forward end, a rear end, and an inner passage between 
        the two ends. The fan is arranged with respect to the barrel to direct an air stream through the inner passage from the rear end to the forward end. The loading chamber is mounted on the 
        barrel and has an opening directed into the inner passage. The loading chamber is sized and shaped to hold a plurality of projectiles and the opening is sized and shaped to sequentially 
        release the plurality of projectiles into the inner passage of the barrel one at a time. The trigger is electrically connected to the fan. Pulling the trigger causes the fan to drive a 
        large volume of air through the gun barrel and the air stream to accelerate as it travels through a narrow passage of the gun barrel. Projectiles sequentially fall into the air stream one at a time and are quickly released from the gun as the air stream accelerates through the gun barrel and exits the gun barrel.'''

        expected = ['automatic', 'toy', 'gun', 'air', 'air stream', 'fan', 'rapid firing', 'toy gun', 'projectiles',
                    'barrel']
        actual = self.tfidf.extract_popular_ngrams(text)

        self.assertGreaterOrEqualDiceScore(expected, actual)

        # The computer picks up terms such as barrel that are already common to guns, and therefore not required

    def test_patent_US08744823_20140603(self):
        text = '''A method for creating, from a first axisymmetrical surface, a second surface belonging to a sub-system of a complex system, in which the second surface observes at 
        least one constraint, is disclosed. The method includes: modeling the first axisymmetrical surface, while observing the constraints with at least one parameter, the modeling step 
        including a sub-step for discretizing the first axisymmetrical surface in several points, the parameter being a coordinate of one of these points in a reference system associated 
        with at least one portion of this sub-system, and a sub-step for reconstructing the first axisymmetrical surface from the at least one point and from the at least one constraint; 
        modifying the at least one parameter in the reference system for modeling the second surface; and recording the second surface in a memory of the computer.'''

        expected = ['computer', 'memory', 'axis', 'surface', 'reconstruct', 'axisymmetrical surface', 'coordinate']
        actual = self.tfidf.extract_popular_ngrams(text)

        self.assertGreaterOrEqualDiceScore(expected, actual)

        # I think this is about computer disk drive platters, but described using other terms

    def test_patent_US07444853_20081104(self):
        text = '''An impulse event separating method, and an apparatus to perform the method, the method including dividing an input signal into frame units and dividing each frame 
        into a plurality of frequency sub-bands; obtaining a power variation and phase variation of the signal of each of the frequency sub-bands, and detecting a plurality of local 
        onsets using the power variation and the phase variation; obtaining a global onset from the local onsets and triggering a plurality of event components using the local onsets 
        and the global onset; tracking and combining the event components in each of the frequency sub-bands to form events; and determining whether the events comprise an impulse event 
        with reference to an impulse event property.'''

        expected = ['impulse', 'event', 'separat', 'input', 'signal', 'input signal', 'frame', 'frequency', 'band',
                    'sub-band', 'power', 'variation', 'phase', 'onset', 'local', 'global']
        actual = self.tfidf.extract_popular_ngrams(text)

        self.assertGreaterOrEqualDiceScore(expected, actual)

    def test_patent_US08571019_20131029(self):
        text = '''A packet communication address is delivered through in-band signaling over a circuit-switched network connection for a call, so the packet communication address can 
        be used to establish a packet session outside the circuit-switched network. When the call is initiated from an initiating communication client to a terminating communication 
        client using a directory number associated with the terminating communication client, a portion of the bearer path is established through the circuit-switched network between 
        two network gateways. The packet communication address for the originating communication client may be received in a call establishment request by an originating network gateway. 
        The originating network gateway will use in-band signaling over the bearer path to provide the packet communication address for the originating communication client to a terminating 
        network gateway. Upon receipt of the packet communication address, the terminating network gateway will initiate transferring the call to a packet session established outside the 
        circuit-switched network.'''

        expected = ['packet', 'communicat', 'address', 'in-band', 'signaling', 'circuit-switched', 'network', 'call',
                    'gateway', 'network gateway', 'network']
        actual = self.tfidf.extract_popular_ngrams(text)

        self.assertGreaterOrEqualDiceScore(expected, actual)

    def test_patent_US09393703_20160719(self):
        text = '''An umbilical member attachment device including a first umbilical member-use first fastening part and second fastening part respectively fastening a first umbilical 
        member to a first member and second member, and a second umbilical member-use first fastening part and second fastening part respectively fastening a second umbilical member to a 
        first member and second member. These fastening parts are arranged offset in position from each other on a surface of the robot so that the first umbilical member and the second 
        umbilical member do not cross, and at a reference posture, and so that a distance between the first umbilical member-use second fastening part and the second umbilical member-use 
        second fastening part becomes broader than a distance between the first umbilical member-use first fastening part and the second umbilical member-use first fastening part.'''

        expected = ['umbilical', 'umbilical member', 'fastening part', 'reference posture']
        actual = self.tfidf.extract_popular_ngrams(text)

        self.assertGreaterOrEqualDiceScore(expected, actual)

    def test_patent_US07540261_20090602(self):
        text = '''A method and apparatus for accurately dispensing, monitoring, quantifying, and controlling liquids provided to one or more animals. In some embodiments, a standalone 
        apparatus having a local user interface is coupled to a liquid source and drinking assembly to monitor, quantify, and control the liquid consumption of one or more specific animals. 
        In other embodiments, multiple standalone apparatuses are each coupled to a liquid source and a respective drinking assembly to monitor, quantify, and control the liquid consumed 
        through the respective drinking assembly. In this scenario, each of the standalone apparatuses shares one or more common remote user interface panels. In yet another embodiment, 
        standalone apparatuses are networked to other standalone apparatuses. User workstations and central control panels resident on the watering apparatus network, or a third-party 
        network interfaced to the network, allow liquid consumption to be monitored, controlled, and quantified both locally and remotely.'''
        expected = ['liquids', 'monitoring', 'dispensing', 'drinking assembly', 'animals']
        actual = self.tfidf.extract_popular_ngrams(text)

        self.assertGreaterOrEqualDiceScore(expected, actual)

    def test_patent_US09690752_20170627(self):
        text = '''During data resampling, bad samples are ignored or replaced with some combination of the good sample values in the neighborhood being processed. 
        The sample replacement can be performed using a number of approaches, including serial and parallel implementations, such as branch-based implementations, matrix-based 
        implementations, and function table-based implementations, and can use a number of modes, such as nearest neighbor, bilinear and cubic convolution.'''
        expected = ['data resampling', 'samples', 'parallel implementations', 'matrix-based', 'nearest neighbour',
                    'bilinear', 'cubic convolution']
        actual = self.tfidf.extract_popular_ngrams(text)

        self.assertGreaterOrEqualDiceScore(expected, actual)

    def test_patent_US07077241_20060718(self):
        text = '''In an elevator installation with an elevator car and a counterweight suspended and driven by several flat-belt-type suspension devices arranged in parallel, 
        the suspension devices are arranged in parallel vertical planes that run diagonal to main horizontal axes of the counterweight and/or of the elevator car. Mounted on the 
        counterweight and on the elevator car are suspension-sheave systems of which at least one comprises several suspension-sheave units which each have one suspension sheave and 
        are arranged adjacent to each other, the suspension-sheave units being fastened to the counterweight and/or to the elevator car in such manner that the axles of the suspension 
        sheaves are essentially horizontal and each are swivelable about one associated vertical axis.'''

        expected = ['elevator', 'elevator car', 'flat-belt-type', 'suspension', 'suspension devices',
                    'parallel vertical planes', 'horizontal axes', 'suspension-sheave', 'axles']
        actual = self.tfidf.extract_popular_ngrams(text)

        self.assertGreaterOrEqualDiceScore(expected, actual)

    def test_patent_US08514680_20130820(self):
        text = '''A recording device including a light irradiating and light receiving unit which, in regard to an optical recording medium, which has a reference surface and a recording 
        layer which is formed at a depth position different from the reference surface, where a pit row where intervals of pit formable positions in one circumference is limited to a first 
        interval is formed in the reference surface, and which has a plurality of pit row phases by intervals of the pit formable positions in a pit row formation direction being set in a 
        position which is each deviated by a predetermined second interval in the pit row which are arranged in the radial direction, irradiates a first light as recording light with regard 
        to the recording layer and a second light for obtaining reflected light from the reference surface and which receives reflected light of the second light from the reference surface.'''

        expected = ['recording device', 'optical', 'circumference', 'radial direction', 'recording light',
                    'reflected light', 'reference surface']
        actual = self.tfidf.extract_popular_ngrams(text)

        self.assertGreaterOrEqualDiceScore(expected, actual)

    def test_patent_US09040602_20150526(self):
        text = '''The invention relates to a dental material which comprises a polymerizable bisphosphonic acid of Formula I: The invention also relates to the use of a polymerizable 
        bisphosphonic acid of Formula I for the preparation of a dental material and in particular for the preparation of an adhesive, cement or composite.'''

        expected = ['dental', 'polymerizable', 'bisphosphonic', 'acid', 'bisphosphonic acid', 'adhesive', 'cement',
                    'composite']
        actual = self.tfidf.extract_popular_ngrams(text)

        self.assertGreaterOrEqualDiceScore(expected, actual)

    def test_patent_US08020436_20110920(self):
        text = '''Embodiments of the present invention include a fiber optic seismic sensing system for permanent downhole installation. In one aspect, the present invention 
        includes a multi-station, multi-component system for conducting seismic reservoir imaging and monitoring in a well. Permanent seismic surveys may be conducted with embodiments 
        of the present invention, including time-lapse (4D) vertical seismic profiling (VSP) and extended micro-seismic monitoring. Embodiments of the present invention provide the ability 
        to map fluid contacts in the reservoir using 4D VSP and to correlate micro-seismic events to gas injection and production activity.'''

        expected = ['fiber', 'optic', 'fiber optic', 'seismic', 'sens', 'downhole', 'reservoir', 'imaging', 'survey',
                    'profil', 'micro-seismic', 'monitor', 'fluid', 'VSP']
        actual = self.tfidf.extract_popular_ngrams(text)

        self.assertGreaterOrEqualDiceScore(expected, actual)

    def test_patent_US07325354_20080205(self):
        text = '''A weapon aiming system may utilize a laser diode and a reflective coating on an optical element to generate a 
        red dot aim point for a shooter with a bright view to the target with minimal color distortion. The optical element may
        utilize an off-axis parabolic lens to reduce parallax to improve sighting accuracy. The weapon aiming system may utilize 
        visible and infrared aim lasers that are coaligned to simplify boresighting of the weapon and to simplify target acquisition. 
        The weapon aiming system may include a magnifier and a sight being disposed along a longitudinal rail of a weapon in a position
        with the close quarter combat sight being disposed between the magnifier and the weapon muzzle.'''
        expected = ['weapon', 'laser', 'diode', 'reflective', 'magnifier', 'muzzle', 'coating', 'optical', 'parabolic',
                    'infrared',
                    'boresighting', 'lens', 'parallax', 'sighting accuracy', 'target acquisition', 'laser diode',
                    'reflective coating',
                    'parabolic lens', 'aim', 'color distortion', 'element', 'target', 'optical element']
        actual = self.tfidf.extract_popular_ngrams(text)

        self.assertGreaterOrEqualDiceScore(expected, actual)

    def test_patent_US08311860_20121113(self):
        text = '''A method of determining a future market scenario for an industry that includes obtaining subjective 
        data from experts in an industry, combining the subjective data and determining from the combined subjective 
        data which market scenario will apply to the industry in the future. '''
        expected = ['industry', 'market', 'subjective data', 'data', 'market scenario']
        actual = self.tfidf.extract_popular_ngrams(text)

        self.assertGreaterOrEqualDiceScore(expected, actual)

    def test_patent_US08002101_20110823(self):
        text = '''A torque based park lock assembly for motor vehicles, especially passenger cars and trucks equipped 
        with automatic transmissions, includes a lobed wheel secured to a transmission output shaft and a rotation 
        restricting flexible lever which may be selectively moved into engagement with the lobed wheel by a cam. A 
        bi-directional motor rotates the cam. A return spring translates the lever away from the lobed wheel when the 
        cam is rotated into a disengaged (non-Park) position. '''
        expected = ['torque', 'park lock', 'lock', 'lock assembly', 'motor', 'vehicles', 'motor vehicles', 'cars',
                    'trucks', 'automatic',
                    'automatic transmissions', 'transmissions', 'shaft', 'transmission shaft', 'lever',
                    'flexible lever',
                    'wheel', 'cam', 'bi-directional motor', 'motor']
        actual = self.tfidf.extract_popular_ngrams(text)

        self.assertGreaterOrEqualDiceScore(expected, actual)

    def test_patent_US09924130_20180320(self):
        text = '''Computer implemented techniques that involve captured, e.g., self-captured video for educational 
        and other uses such as improving job performance of geographically distributed employees and incremental 
        video optimizations and compressions are described. '''
        expected = ['computer', 'self-captured video', 'video', 'educational', 'job performance', 'job', 'employees',
                    'video optimizations',
                    'optimizations', 'compressions']
        actual = self.tfidf.extract_popular_ngrams(text)

        self.assertGreaterOrEqualDiceScore(expected, actual)

    def test_patent_US07875029_20110125(self):
        text = '''One example of surgical apparatus for treating tissue may include an effector including at least 
        two jaws movable toward one another, where the effector holds and is configured to deploy a of clips in a 
        clip application mode, and where the jaws are configured to deliver energy to coagulate tissue in a 
        coagulation mode, where said effector is switchable between clip application mode and coagulation mode. 
        Another example of surgical apparatus for treating tissue may include an effector holding clips, 
        and two fingers movable toward one another to close the clips one at a time, where each finger is a different 
        pole of a bipolar coagulator. An exemplary method for treating tissue with a surgical apparatus may include 
        placing the surgical apparatus adjacent to tissue at a location, selecting one of a plurality of operational 
        modes of the surgical apparatus, where the operational modes include clip application mode and coagulation 
        mode, and actuating the surgical apparatus according to the selected operational mode. '''
        expected = ['surgical', 'apparatus', 'surgical apparatus', 'tissue', 'treating', 'effector', 'jaws',
                    'clip application mode',
                    'clip', 'coagulate tissue', 'coagulation mode', 'bipolar coagulator']
        actual = self.tfidf.extract_popular_ngrams(text)

        self.assertGreaterOrEqualDiceScore(expected, actual)

    def test_patent_US08854956_20141007(self):
        text = '''An apparatus comprising a path computation element (PCE) associated with a domain in a network and 
        configured to find a segment of a Multiprotocol Label Switching (MPLS) Traffic Engineering (TE) Label 
        Switched Path (LSP) that crosses a plurality of domains in the network using a Constraint Shortest Path First 
        (CSPF) algorithm or a reverse CSPF algorithm that computes a plurality of shortest paths in the domain of 
        which the segment is selected, wherein the CSPF algorithm or the reverse CSPF algorithm is selected to reduce 
        the number of shortest path computations in the domain based on the number of starting nodes and ending nodes 
        that are considered for computing the shortest paths in the domain. '''
        expected = ['path computation element', 'computation', 'apparatus', 'Multiprotocol Label Switching',
                    'multiprotocol',
                    'Traffic Engineering', 'traffic', 'engineering', 'Label Switched Path', 'plurality', 'domains',
                    'network',
                    'constraint', 'algorithm', 'starting nodes', 'nodes', 'ending nodes']
        actual = self.tfidf.extract_popular_ngrams(text)

        self.assertGreaterOrEqualDiceScore(expected, actual)

    def test_patent_US08026129_20110927(self):
        text = '''A stacked integrated circuit package system is provided forming a first stack layer having a first 
        integrated circuit die on a first substrate, forming a second stack layer having a second integrated circuit 
        die on a second substrate, and mechanically and electrically connecting a spacer layer having a first passive 
        component between the second stack layer and the first stack layer. '''
        expected = ['integrated circuit package', 'stack', 'integrated circuit', 'circuit', 'circuit package system',
                    'system',
                    'substrate', 'mechanically', 'electrically', 'passive component', 'passive', 'layer stack']
        actual = self.tfidf.extract_popular_ngrams(text)

        self.assertGreaterOrEqualDiceScore(expected, actual)

    def test_patent_US08837251_20140916(self):
        text = '''An array configuration capable of supplying a necessary and sufficient current in a small area is 
        achieved and a reference cell configuration suitable to temperature characteristics of a TMR element is 
        achieved. In a memory using inversion of spin transfer switching, a plurality of program drivers are arranged 
        separately along one global bit line, and one sense amplifier is provided to one global bit line. A reference 
        cell to which “1” and “0” are programmed is shared by two arrays and a sense amplifier. '''
        expected = ['array', 'current', 'reference cell', 'cell', 'temperature', 'element', 'inversion',
                    'spin transfer', 'plurality',
                    'program drivers', 'drivers', 'sense amplifier', 'amplifier']
        actual = self.tfidf.extract_popular_ngrams(text)

        self.assertGreaterOrEqualDiceScore(expected, actual)

    def test_patent_US09677958_20170613(self):
        text = '''A preload device has a receiving pocket for receiving a force sensor. A tool engagement section is 
        spaced apart from the receiving pocket in the direction of the longitudinal axis. At least one elastically 
        flexible section is non-detachably arranged between the receiving pocket and at least one of at least two 
        force introduction plates. Elastic bending of the at least one elastically flexible section through the 
        application of forces to at least one of the force introduction plates and to the tool engagement section 
        effects a reduction in the height of a relaxed state of the preload device to facilitate insertion of the 
        preload device in a recess in a machine part or between multiple machine parts or in a drum. '''
        expected = ['preload device', 'device', 'receiving pocket', 'pocket', 'force sensor', 'force', 'sensor',
                    'elastically flexible section', 'flexible section', 'introduction plates', 'recess', 'machine',
                    'machine parts', 'drum',
                    'plate', 'tool']
        actual = self.tfidf.extract_popular_ngrams(text)

        self.assertGreaterOrEqualDiceScore(expected, actual)

    def test_patent_US09231600_20160105(self):
        text = '''A number of methods and clock generator units are disclosed to produce low Phase Noise clocks for 
        use in Radio Frequency systems. The methods and clock generator units all use two reference clocks: a 
        frequency-accurate reference that has comparatively high Phase Noise, and a frequency-inaccurate reference 
        such as that from a BAW or MEMS clock source that has comparatively low Phase Noise. By combining multiple 
        Phase-Locked Loops and a mixer, it is possible to produce flexible output frequencies whose frequency 
        accuracy is derived from the first reference clock but whose Phase Noise level is derived from the second 
        reference clock, all in a readily-integrated and relatively low-cost system. '''
        expected = ['clock', 'generator', 'clock generator', 'Phase Noise clocks', 'phase', 'noise', 'Radio Frequency',
                    'radio',
                    'frequency', 'mixer', 'system', 'generator units', 'unit', 'product']
        actual = self.tfidf.extract_popular_ngrams(text)

        self.assertGreaterOrEqualDiceScore(expected, actual)

    def test_patent_US09601171_20170321(self):
        text = '''A storage device includes a nonvolatile memory, and a memory controller adapted to control the 
        nonvolatile memory and to transmit a first timing signal to the nonvolatile memory at a read operation. The 
        nonvolatile memory includes a nonvolatile memory device adapted to output read data and a second timing 
        signal in response to the first timing signal, and a retiming circuit adapted to app_detect a locking delay 
        according to the first timing signal, to produce a third timing signal from the second timing signal using 
        the detected locking delay, to retime the read data by latching the read data in synchronization with the 
        third timing signal and to output the third timing signal and the retimed read data to the memory controller. 
        '''
        expected = ['storage device', 'storage', 'device', 'nonvolatile memory', 'nonvolatile', 'memory',
                    'memory controller',
                    'controller', 'transmit', 'signal', 'circuit', 'locking delay', 'synchronization', 'read data',
                    'time']
        actual = self.tfidf.extract_popular_ngrams(text)

        self.assertGreaterOrEqualDiceScore(expected, actual)

    def test_patent_US09394705_20160719(self):
        text = '''Movable centring for platforms in ports which comprises a mobile structure ( 1 ) with various 
        formworks ( 2 ) and is supported on piles ( 3 ). The support of the mobile structure ( 1 ) on the piles ( 3 ) 
        is carried out by means of, at least, two supporting devices ( 5 ), wherein each supporting device ( 5 ) 
        comprises one front supporting beam ( 6 d ), which comprises at least two coupling pieces ( 18 ), 
        one rear supporting beam ( 6 t ) which comprises at least two coupling pieces ( 18 ) and a lattice ( 7 ), 
        beneath each supporting beam ( 6 d,  6 t ). The two supporting beams ( 6 d,  6 t ) are joined together by 
        means of the at least two coupling pieces ( 18 ), thus leaving the supporting device ( 5 ) fixed to the pile 
        ( 3 ) by means of this union, and joining each supporting beam ( 6 d,  6 t ) to the mobile structure ( 1 ) by 
        means of, at least, two vertical latching bars ( 8 ). '''
        expected = ['platforms', 'ports', 'mobile structure', 'formworks', 'piles', 'supporting device', 'coupling',
                    'lattice',
                    'supporting beam', 'latching bars', 'mobile']
        actual = self.tfidf.extract_popular_ngrams(text)

        self.assertGreaterOrEqualDiceScore(expected, actual)

    def test_patent_US09589149_20170307(self):
        text = '''A profiling service may determine, local to a device, user profile attributes associated with a 
        device user based on interaction of the device user with the device, based on device-local monitoring of 
        device user interactions with the device, and may store the user profile attributes in a memory. The 
        profiling service may be configured as an augmentation to a device operating system of the device. A profile 
        exposure component may manage exposure of information associated with the user profile attributes to 
        applications operating locally on the device, without exposure to the applications or to third parties of 
        information determined as sensitive to the device user. '''
        expected = ['profiling service', 'service', 'device', 'user profile', 'user', 'profile', 'attributes', 'memory',
                    'augmentation',
                    'operating system', 'system', 'exposure component', 'exposure', 'component', 'applications',
                    'third parties',
                    'sensitive', 'local device', 'device user']
        actual = self.tfidf.extract_popular_ngrams(text)

        self.assertGreaterOrEqualDiceScore(expected, actual)

    def test_patent_US07708361_20100504(self):
        text = '''The inkjet recording apparatus initially deposits ink on a front surface of a recording medium and 
        subsequently deposits ink on a rear surface of the recording medium in such a manner that images are formed 
        by the ink on the front surface and the rear surface of the recording medium, wherein, when time after 
        depositing the ink on a first region of the front surface of the recording medium until depositing the ink on 
        a second region of the rear surface of the recording medium which corresponds to the first region is taken to 
        be ΔT 1  (sec) and viscosity of the ink to be deposited on the first region and the second region of the 
        recording medium is taken to be η (mPa·sec), then the inkjet recording apparatus deposits the ink on the 
        first region and the second region in such a manner that a following relationship is satisfied: 

            Δ T 1<0.45×η.'''
        expected = ['inkjet', 'recording apparatus', 'apparatus', 'ink', 'images', 'viscosity', 'recording medium',
                    'medium',
                    'surface']
        actual = self.tfidf.extract_popular_ngrams(text)

        self.assertGreaterOrEqualDiceScore(expected, actual)

    def test_patent_US06942397_20050913(self):
        text = '''A packaged fiber-coupled optical device comprises an alignment housing with a fiber retainer, 
        optical fiber segment(s), and optical component(s) (on substrate(s) with fiber groove(s)). Upon assembly the 
        protruding end(s) of the fiber segment(s) is/are positioned against the fiber retainer, and the fiber groove(
        s) is/are aligned with the protruding end(s) of the fiber segment(s). The fiber retainer urges the protruding 
        end(s) of the fiber segment(s) into the fiber groove(s). The fiber groove(s) position the protruding end(s) 
        of the optical fiber(s) seated therein for optical coupling with optical component(s). The alignment housing 
        and/or a fiber subassembly may be configured for engaging a mating fiber-optic connector. '''
        expected = ['fiber', 'fiber-coupled', 'optical', 'device', 'optical device', 'alignment housing', 'housing',
                    'retainer', 'segment', 'optical component', 'component', 'fiber groove', 'groove', 'connector',
                    'alignment']
        actual = self.tfidf.extract_popular_ngrams(text)

        self.assertGreaterOrEqualDiceScore(expected, actual)

    def test_patent_US07633944_20091215(self):
        text = '''Techniques are described for managing timeouts of filter criteria in a packet flow capture 
        applications. The techniques allow for handling large amounts of timeouts used when monitoring a high volume 
        of packet flows, without placing extreme demands on the operating system for managing the timeouts. The 
        timeout data structure may be a circular array having a plurality of elements. The timeout array represents a 
        span of time and the elements represent sequential units of time. Each element contains one or more pointers. 
        The pointer may point to an entry in the filter table, or may be a null pointer. A timer thread periodically 
        checks the timeout array to determine whether any timeouts occur at the current time. The timer thread checks 
        the element of the array corresponding to the current time by computing an index into the array based on the 
        current time. '''
        expected = ['managing', 'timeouts', 'filter', 'criteria', 'packet', 'flow', 'flow capture', 'capture',
                    'applications',
                    'monitoring', 'packet flows', 'operating system', 'operating', 'system', 'circular array', 'array',
                    'elements',
                    'time', 'pointers', 'computing', 'index', 'technique']
        actual = self.tfidf.extract_popular_ngrams(text)

        self.assertGreaterOrEqualDiceScore(expected, actual)

    def test_patent_US08331547_20121211(self):
        text = '''The present invention relates to an apparatus ( 10 ) for handling telephone calls which comprises: 
        means for configuring the operation of the apparatus based on user preferences relating to the handling of 
        calls from particular callers, means for storing the user preferences, means for processing incoming calls 
        based on the user preferences, and means for receiving updates to the user preferences in dependence on 
        changes to a preference database for storing the preferences of at least one user. The invention also relates 
        to a telecommunications system, and to an associated server. '''
        expected = ['apparatus', 'telephone calls', 'telephone', 'calls', 'operation', 'user preferences', 'user',
                    'incoming calls', 'incoming', 'database', 'telecommunications system', 'telecommunications',
                    'system', 'server', 'caller']
        actual = self.tfidf.extract_popular_ngrams(text)

        self.assertGreaterOrEqualDiceScore(expected, actual)

    def test_patent_US07346451_20080318(self):
        text = '''When a guiding route is included in a display target region at a photograph display mode, 
        route data of the guiding route is also read. Within the read route data, information of a region for editing 
        a photograph is stored. This information includes a road kind (color information), coordinates of the 
        start/end points of the road, and a road width when editing is for a road. Next, photograph data is read, 
        so that a color of the photograph data is edited based on the stored information of the route data. For 
        instance, a color of the photograph data in a corresponding road region is changed to red that represents 
        “national road.” '''
        expected = ['guiding', 'display', 'display target', 'target', 'photograph', 'display', 'route', 'data', 'road',
                    'color', 'coordinates', 'points', 'region', 'route data']
        actual = self.tfidf.extract_popular_ngrams(text)

        self.assertGreaterOrEqualDiceScore(expected, actual)

    def test_patent_US07267882_20070911(self):
        text = '''An improved ceramic/metal composite material is disclosed which is fully reacted with aluminium. 
        The composite is made from a ceramic preform, such as silicon carbide, having a binding agent, 
        such as silica, that is contacted with a metal mixture or alloy, such as aluminium/silicon, that reacts with 
        the binding agent to form a ceramic/metal composite material. Also disclosed is a method of making the 
        improved composite material and articles made incorporating the material. '''
        expected = ['ceramic', 'metal', 'composite', 'composite material', 'material', 'reacted', 'aluminium',
                    'silicon carbide',
                    'binding', 'agent', 'binding agent', 'silica', 'alloy']
        actual = self.tfidf.extract_popular_ngrams(text)

        self.assertGreaterOrEqualDiceScore(expected, actual)


if __name__ == '__main__':
    unittest.main()
