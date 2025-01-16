import pandas as pd
import gandula
from dotenv import load_dotenv
import os

load_dotenv()

def events_to_df(events, match_id):
    """
    Converte uma lista de 'events' em um DataFrame com colunas:
      match_id, event_id, possession_id, possession_type, 
      player_id, receiver, outcome, carry_type
    """

    rows = []

    for event in events:
        # Aqui assumimos que existe 'event.id' e 'event.game.id' (ou algo similar),
        # para identificar o match_id e event_id. Ajuste conforme seu modelo real.
        event_id = event.id if event.id else None
        team_id = event.team.id if event.team else None

        # Itera sobre cada posse (possessionEvent) dentro de event
        for possessionEvent in event.possessionEvents:
            possession_id = possessionEvent.id

            # Se for um PASS
            if possessionEvent.passingEvent:
                pass_event = possessionEvent.passingEvent
                passer = pass_event.passerPlayer.id if pass_event.passerPlayer else None
                receiver = pass_event.receiverPlayer.id if pass_event.receiverPlayer else None
                outcome = pass_event.passOutcomeType.value if pass_event.passOutcomeType else None
                
                row = {
                    "match_id": match_id,
                    "team_id": team_id,
                    "event_id": event_id,
                    "possession_id": possession_id,
                    "possession_type": "PASS",
                    "player_id": passer,
                    "receiver": receiver,
                    "outcome": outcome,
                    "carry_type": None
                }
                rows.append(row)

            # Se for um CARRY
            elif possessionEvent.ballCarryEvent:
                carry_event = possessionEvent.ballCarryEvent
                carrier = carry_event.ballCarrierPlayer.id if carry_event.ballCarrierPlayer else None
                dribble_outcome = carry_event.dribbleOutcomeType.value if carry_event.dribbleOutcomeType else None
                carry_type = carry_event.ballCarryType.value if carry_event.ballCarryType else None

                row = {
                    "match_id": match_id,
                    "event_id": event_id,
                    "possession_id": possession_id,
                    "possession_type": "CARRY",
                    "player_id": carrier,
                    "receiver": None,
                    "outcome": dribble_outcome,
                    "carry_type": carry_type
                }
                rows.append(row)

            # Se quiser ignorar outros tipos de eventos, basta não fazer nada

    # Converte a lista de dicionários em um DataFrame
    df = pd.DataFrame(rows, columns=[
        "match_id",
        "event_id",
        "possession_id",
        "possession_type",
        "player_id",
        "receiver",
        "outcome",
        "carry_type"
    ])

    return df

def get_match_events(match_id):
    api_url = os.getenv('api_url')
    api_key = os.getenv('api_key')
    try:
        events = gandula.get_match_events(
            match_id=match_id, api_url=api_url, api_key=api_key
        )
        return events_to_df(events, match_id)
    except:
        print(f"Error processing match_id {match_id}")
        return pd.DataFrame()