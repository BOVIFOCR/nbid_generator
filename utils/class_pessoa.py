import random
import string
from datetime import datetime, timedelta

import pandas as pd

# DICIONÁRIO DE PALAVRAS

# TODO: FIXO obs e tipo_h, cargo, aspa, doar e org.
tipo_h = {0: "A", 1: "B", 2: "C", 3: "D", 4: "E", 5: "AB"}

fator_rh = {0: "A+", 1: "A-", 2: "B+", 3: "B-", 4: "O+", 5: "O-", 6: "AB+", 7: "AB-"}

obs = {
    0: "EXERCE ATIVIDADE REMUNERADA;",
    1: "A;",
    2: "B;",
    3: "C;",
    4: "D;",
    5: "E;",
    6: "F;",
    7: "G;",
    8: "H;",
    9: "I;",
    10: "J;",
    11: "K;",
    12: "L;",
    13: "M;",
    14: "N;",
    15: "O;",
    16: "P;",
    17: "Q;",
    18: "R;",
    19: "S;",
    20: "T;",
    21: "U;",
    22: "V;",
    23: "W;",
    24: "X;",
    25: "Y;",
    26: "Z;",
}

ufs = [
    'AC',
    'AL',
    'AM',
    'AP',
    'BA',
    'CE',
    'DF',
    'ES',
    'GO',
    'MA',
    'MT',
    'MS',
    'MG',
    'PA',
    'PB',
    'PR',
    'PE',
    'PI',
    'RJ',
    'RS',
    'RN',
    'RO',
    'RR',
    'SC',
    'SP',
    'SE',
    'TO'
]

cargo = {0: "DIRETOR", 1: "COORDENADOR", 2: "PRESIDENTE"}

aspa = {0: "COM AVRB VIUVEZ", 1: "COM AVRB DIVÓRCIO"}

org = {
    0: "SDS",
    1: "SSP",
    2: "POM",
    3: "SNJ",
    4: "SPTC",
    5: "SESP",
    6: "SJS",
    7: "POF",
    8: "SES",
    9: "SEJ",
}
# ------------------------------------------------------------------------------------------------------------------

def gen_rand_datetime(min_time=datetime(1950,1,1,00,00,00), max_time=datetime.now()):
    # Get total number of days between min and max time
    total_days = (max_time - min_time).days
    
    # Generate random number of days in between
    rand_days = random.randint(1, total_days-1)

    # add rand_days to min date, generating rand date between min and max
    new_time = min_time + timedelta(days=rand_days)
    return new_time

def time_to_str(timestr):
    ret = timestr.strftime("%d/%m/%Y")
    return ret

def make_rg():
    def formata_rg(n_rg):
        rg_f = ""
        for i in range(len(n_rg)):
            rg_f = rg_f + n_rg[i]
            if i in [1, 4]:
                rg_f = rg_f + "."
            elif i == 7:
                rg_f = rg_f + "-"
        return rg_f

    def verf_rg(n_rg):
        dig_v = str(random.randint(0, 9))
        int_values = []
        peso = 2
        for i in range(len(n_rg)):
            int_values.append(int(n_rg[i]) * (peso + i))
        soma = sum(int_values)
        for x in range(10):
            result = soma + x * 100
            if result % 11 == 0:
                dig_v = str(x)
                break
        return dig_v

    seq_rg = ""
    for _ in range(8):
        random.seed()
        sel_num = random.randint(0, 9)
        seq_rg = seq_rg + str(sel_num)
    dig = verf_rg(seq_rg)
    seq_rg = seq_rg + dig
    seq_rg = formata_rg(seq_rg)
    return seq_rg

def make_name():
    file = open("./files/nome.txt", "r", encoding="ISO-8859-1")
    names = file.readlines()
    full_name = ""
    for _ in range(3):
        random.seed()
        sel_num = random.randint(0, len(names) - 1)
        full_name = full_name + names[sel_num].rstrip("\n") + ""
    full_name = full_name[:-1]
    return full_name



class Person:
    def __init__(self):

        self.entities = {}

        self.set_nome()
        self.set_filiacao(1)
        self.set_filiacao(2)
        self.set_cpf()
        self.set_rg()
        self.set_cnh()

        self.set_pis()
        self.set_dni()
        self.set_cid_est(50)
        self.set_est()
        self.set_cid(50)
        self.set_d_orig()
        self.set_datanasc()
        self.set_dataexp()
        self.set_org()
        self.set_obs()
        self.set_cns()
        self.set_fator_rh()
        self.set_titulo()
        self.set_militar()
        self.set_profissional()
        self.set_ctps()
        self.set_serie()

    def get_entity(self, tipo):
        if tipo in self.entities:
            if tipo in ('datanasc', 'dataexp'):
                return time_to_str(self.entities[tipo])
            return self.entities[tipo]
        else:
            return None

    def set_nome(self):
        self.entities['nome'] = make_name()
        return self.entities['nome']

    def set_filiacao(self, num=1):
        full_name = make_name()
        if num == 1:
            self.entities['filiacao1'] = full_name
        else:
            self.entities['filiacao2'] = full_name
        return full_name

    def set_cpf(self):
        def formata_cpf(n_cpf):
            formatado = ""
            for i in range(len(n_cpf)):
                formatado = formatado + n_cpf[i]
                if i in [2, 5]:
                    formatado = formatado + "."
                elif i == 8:
                    formatado = formatado + "-"
            return formatado

        def dig_verificador(n_cpf):
            int_values = []
            peso = 10 if len(n_cpf) == 9 else 11
            for i in range(len(n_cpf)):
                int_values.append(int(n_cpf[i]) * (peso - i))
            soma = sum(int_values)
            resto = soma % 11
            dig = "0" if resto in [0, 1] else str(11 - resto)
            return dig

        def make_cpf():
            seq_cpf = ""
            for _ in range(9):
                random.seed()
                sel_num = random.randint(0, 9)
                seq_cpf = seq_cpf + str(sel_num)
            f_dig = dig_verificador(seq_cpf)
            seq_cpf = seq_cpf + f_dig
            s_dig = dig_verificador(seq_cpf)
            seq_cpf = seq_cpf + s_dig
            seq_cpf = formata_cpf(seq_cpf)
            return seq_cpf

        r_cpf = make_cpf()
        self.entities['cpf'] = r_cpf
        return r_cpf

    def set_rg(self):
        r_rg = make_rg()
        self.entities['rg'] = r_rg
        return r_rg

    def set_pis(self):
        pis_pasep = ""
        for x in range(11):
            random.seed()
            sel_num = random.randint(0, 9)
            pis_pasep = pis_pasep + str(sel_num)
            if x == 3 or x == 8:
                pis_pasep = pis_pasep + "."
            elif x == 9:
                pis_pasep = pis_pasep + "-"
        # if len(pis_pasep) > qtd_chars:
        #     pis_pasep = pis_pasep[:qtd_chars]
        self.entities['pis'] = pis_pasep
        return pis_pasep

    def set_dni(self):
        dni = "*****"
        self.entities['dni'] = dni
        return dni

    def set_est(self):
        random.seed()
        df = pd.read_csv(r"./files/cid_est.csv", encoding="utf-8")
        sel_num = random.randint(0, df.shape[0] - 1)

        sel_est = df["UF"][sel_num].upper()
        self.entities['uf'] = sel_est
        return sel_est

    def set_cid(self, qtd_chars):
        df = pd.read_csv(r"./files/cid_est.csv", encoding="utf-8")
        df_filter = (
            df.loc[df["Município"].apply(lambda x: len(x) < qtd_chars)]
        ).dropna(how="all")

        if df_filter.shape[0] > 0:
            random.seed()
            sel_num = random.randint(0, df_filter.shape[0] - 1)
            sel_cid = df_filter["Município"].values[sel_num].upper()
        else:
            sel_cid = "ITU"

        self.entities['cid'] = sel_cid
        return sel_cid

    def set_cid_est(self, qtd_chars):
        df = pd.read_csv(r"./files/cid_est.csv", encoding="utf-8")
        df_filter = (
            df.loc[df["Município"].apply(lambda x: len(x) < qtd_chars - 4)]
        ).dropna(how="all")

        if df_filter.shape[0] > 0:
            random.seed()
            sel_num = random.randint(0, df_filter.shape[0] - 1)
            sel_est = df_filter["UF"].values[sel_num].upper()
            sel_cid = df_filter["Município"].values[sel_num].upper()
            sel_local = sel_cid + "-" + sel_est
        else:
            sel_local = "ITU-SP"

        self.entities['naturalidade'] = sel_local
        return sel_local

    # TODO: Checar get_cid
    def set_d_orig(self):
        df = pd.read_csv(r"./files/cid_est.csv", encoding="utf-8")
        random.seed()
        sel_num = random.randint(0, len(df) - 1)
        sel_cid = df["Município"][sel_num].upper()
        sel_est = df["UF"].values[sel_num].upper()

        doc = "CMC= " + sel_cid + "-" + sel_est + " ,SEDE"
        self.entities['regcivil'] = doc
        return doc

    def set_datanasc(self):
        if 'dataexp' not in self.entities:
            unf_date = gen_rand_datetime()
        else:
            unf_date = gen_rand_datetime(min_time=self.entities['dataexp'])
        self.entities['datanasc'] = unf_date
        return time_to_str(unf_date)

    def set_dataexp(self):
        if 'datanasc' not in self.entities:
            unf_date = gen_rand_datetime()
        else:
            unf_date = gen_rand_datetime(max_time=self.entities['datanasc'])
        self.entities['dataexp'] = unf_date
        return time_to_str(unf_date)

    def set_org(self):
        random.seed()
        sel_num = random.randint(0, len(org) - 1)
        text = org[sel_num]
        self.entities['orgaoexp'] = text
        return text

    def set_obs(self):
        text = ""
        random.seed()
        sel_num = random.randint(0, len(obs) - 1)
        text = obs[sel_num]
        self.entities['obs'] = text
        return text
    
    def set_fator_rh(self):
        rn = random.randint(0, len(fator_rh.keys())-1)
        rh = fator_rh[rn]
        self.entities['rh'] = rh
        return rh

    def set_titulo(self):
        # 1 out of 10
        if 'datanasc' not in self.entities or \
                    ((datetime.now() - self.entities['datanasc']).days//365) < 16 \
                             or random.randint(0, 9) == 0:
            titulo = "*****"
        else:
            titulo = "0"
            for i in range(0, 12):
                titulo += str(random.randint(0, 9))
        self.entities['te'] = titulo
        return titulo

    def set_ctps(self):
        # 1 out of 10
        if 'datanasc' not in self.entities or \
                ((datetime.now() - self.entities['datanasc']).days//365) < 14 \
                    or random.randint(0, 9) == 0 or 'cpf' not in self.entities:
            ctps = "*****"
        else:
            cpf = self.entities['cpf']
            ctps = cpf[0:3] + cpf[4:7] + cpf[8]
        self.entities['ctps'] = ctps
        return ctps

    def set_serie(self):
        if not 'ctps' not in self.entities or 'cpf' not in self.entities:
            serie = "*****"
        else:
            cpf = self.entities['cpf']
            serie = cpf[-5:-3] + cpf[-2:]
        self.entities['serie'] = serie
        return serie

    def set_cns(self):
        if random.randint(0, 5) != 5: # 4 out of 5
            cns = "*****"
        else:
            cns = ""
            for i in range(0, 15):
                cns += str(random.randint(0, 9))
        self.entities['cns'] = cns
        return cns

    def set_profissional(self):
        if random.randint(0, 19) != 00: # 19 out of 20
            profissional = "*****"
        else:
            if random.randint(0, 1) == 1:
                preamb = "CREA"
            else:
                preamb = "CONFEA"
            preamb = preamb + "/" + ufs[random.randint(0,len(ufs)-1)]

            profissional = preamb + " 0"
            for i in range(0, 7):
                profissional += str(random.randint(0, 9))
                if i in (0, 3):
                    profissional += "."
        self.entities['profissional'] = profissional 
        return profissional

    def set_militar(self):
        if random.randint(0, 9) != 0:
            militar = "*****"
        else:
            militar = ""
            for i in range(0, 7):
                militar += str(random.randint(0, 9))
        self.entities['militar'] = militar
        return militar

    def set_cnh(self):
        cnh = make_rg()
        self.entities['cnh'] = cnh
        return cnh
